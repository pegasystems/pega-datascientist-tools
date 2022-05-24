import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Set, Union
import pydot
from IPython.display import Image, display
import functools
import operator
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statistics import mean
from math import exp
import requests
import urllib.request


class ADMTrees:
    """Functions for ADM Gradient boosting

    ADM Gradient boosting models consist of multiple trees,
    which build upon each other in a 'boosting' fashion.
    This class provides some functions to extract data from
    these trees, such as the features on which the trees split,
    important values for these features, statistics about the trees,
    or visualising each individual tree.

    Parameters
    ----------
    file: str
        The input file as a json (see notes)

    Attributes
    ----------

    trees: Dict
    properties: Dict
    learning_rate: float
    model: Dict
    treeStats: Dict
    splitsPerTree: Dict
    gainsPerTree: Dict
    gainsPerSplit: pd.DataFrame
    groupedGainsPerSplit: Dict
    predictors: Set
    allValuesPerSplit: Dict

    Notes
    -----
    The input file is the extracted json file of the 'save model'
    action in Prediction Studio. The Datamart column 'pyModelData'
    also contains this information, but it is compressed and
    the values for each split is encoded. Using the 'save model'
    button, only that data is decompressed and decoded.
    """

    def __init__(self, file: str):
        try:
            self.import_model(file)
        except:
            self.workaround_import(file)
        self.predictors = self.getPredictors()
        self.treeStats = self.getTreeStats()
        (
            self.splitsPerTree,
            self.gainsPerTree,
            self.gainsPerSplit,
        ) = self.getGainsPerSplit()
        self.groupedGainsPerSplit = self.getGroupedGainsPerSplit()
        self.allValuesPerSplit = self.getAllValuesPerSplit()

    def import_model(self, file: str):
        """Imports the 'regular' json file"""
        with open(file) as f:
            self.trees = json.load(f)
        self.properties = {
            prop[0]: prop[1] for prop in self.trees.items() if prop[0] != "model"
        }
        self.learning_rate = self.properties["configuration"]["parameters"][
            "learningRateEta"
        ]
        try:
            self.model = self.trees["model"]["boosters"][0]["trees"]
        except:
            self.model = self.trees["model"]["model"]["boosters"][0]["trees"]

    def workaround_import(self, file: str):
        """Imports the 'old' txt file, with json embedded in it."""

        def separate_txt_sections(f, decode=False):
            """Extracts the relevant sections from the txt file"""
            modelStart = "model=AdaptiveBoostScoringModel{"
            jsonStart = "model={"
            configurationStart = "configuration=GradientBoostModelRuleConfiguration{"
            data = {"model": [], "json": [], "config": [], "other": []}
            current_type = "other"
            for line in f:
                if decode:
                    line = line.decode("utf-8")
                if str(line).startswith(modelStart):
                    current_type = "model"
                elif str(line).startswith(jsonStart):
                    current_type = "json"
                elif str(line).startswith(configurationStart):
                    current_type = "config"
                data[current_type].append(line.strip())
            return data

        try:
            with open(file) as f:
                data = separate_txt_sections(f)
        except FileNotFoundError:
            try:
                response = requests.get(file)
                is_url = True if response.status_code == 200 else False
            except:
                is_url = False
            if is_url:
                data = separate_txt_sections(urllib.request.urlopen(file), decode=True)
            else:
                raise FileNotFoundError(file)

        self.trees = "{" + "".join(data["json"][2:])[:-5]
        self.model = json.loads(self.trees)["trees"]
        self.properties = "".join(data["config"])
        for i in "".join(data["config"]).split(","):
            if i.startswith(" parameters"):
                self.learning_rate = i.split("=")[-1]

    def _depth(self, d: Dict) -> Dict:
        """Calculates the depth of the tree, used in TreeStats."""
        if isinstance(d, dict):
            return 1 + (max(map(self._depth, d.values())) if d else 0)
        return 0

    def parseSplitValues(self, value) -> Tuple[str, str, str]:
        """Parses the raw 'split' string into its three components.

        Once the split is parsed, Python can use it to evaluate.

        Parameters
        ----------
        value: str
            The raw 'split' string

        Returns:
            Tuple[str, str, str]
            The variable on which the split is done,
            The direction of the split (< or 'in')
            The value on which to split
        """

        if isinstance(value, pd.Series):
            value = value["split"]
        if self.nospaces:
            variable, sign, *splitvalue = value.split(" ")
            if sign not in {">", "<", "in", "is", "=="}:
                self.nospaces = False
                variable, sign, *splitvalue = self.parseSplitValuesWithSpaces(value)
        else:
            variable, sign, *splitvalue = self.parseSplitValuesWithSpaces(value)

        if len(splitvalue) == 1 and isinstance(splitvalue, list):
            splitvalue = splitvalue[0]

        if sign in {"<", ">", "=="}:
            splitvalue = float(splitvalue)
        else:
            splitvalue = "".join(splitvalue[1:-1])
            splitvalue = set(splitvalue.split(","))
        return variable, sign, splitvalue

    @staticmethod
    def parseSplitValuesWithSpaces(value) -> Tuple[str, str, str]:
        splittypes = {">", "<", "in", "is", "=="}
        stage = "predictor"
        variable = ""
        sign = ""
        splitvalue = ""

        for item in value.split(" "):
            if item in splittypes:
                stage = "sign"
            if stage == "predictor":
                variable += " " + item
            elif stage == "values":
                splitvalue += item
            elif stage == "sign":
                sign = item
                stage = "values"
        variable = variable.strip()
        return variable, sign, splitvalue

    def getPredictors(self) -> Dict:
        self.nospaces = True
        try:
            predictors = self.properties["configuration"]["predictors"]
        except:
            predictors = []
            for i in self.properties.split("=")[4].split(
                "com.pega.decision.adm.client.PredictorInfo: "
            ):
                if i.startswith("{"):
                    if i.endswith("ihSummaryPredictors"):
                        predictors += [i.split("], ihSummaryPredictors")[0]]
                    else:
                        predictors += [i[:-2]]
        predictorsDict = {}
        for predictor in predictors:
            if isinstance(predictor, str):
                predictor = json.loads(predictor)
            predictorsDict[predictor["name"]] = predictor["type"]
        return predictorsDict

    def getGainsPerSplit(self) -> Tuple[Dict, pd.DataFrame, dict]:
        """Function to compute the gains of each split in each tree."""
        splitsPerTree = {
            treeID: self.getSplitsRecursively(tree=tree, splits=[], gains=[])[0]
            for treeID, tree in enumerate(self.model)
        }
        gainsPerTree = {
            treeID: self.getSplitsRecursively(tree=tree, splits=[], gains=[])[1]
            for treeID, tree in enumerate(self.model)
        }
        splitlist = [value for value in splitsPerTree.values() if value != []]
        gainslist = [value for value in gainsPerTree.values() if value != []]
        total_split_list = functools.reduce(operator.iconcat, splitlist, [])
        total_gains_list = functools.reduce(operator.iconcat, gainslist, [])
        gainsPerSplit = pd.DataFrame(
            list(zip(total_split_list, total_gains_list)), columns=["split", "gains"]
        )
        gainsPerSplit.loc[:, "predictor"] = gainsPerSplit.split.agg(
            lambda x: self.parseSplitValues(x)[0]
        )
        return splitsPerTree, gainsPerTree, gainsPerSplit

    def getGroupedGainsPerSplit(self) -> pd.DataFrame:
        """Function to get the gains per split, grouped by split.

        It adds some additional information, such as the possible values,
        the mean gains, and the number of times the split is performed.
        """
        gainPerSplit = (
            self.gainsPerSplit.groupby("split")
            .agg({"gains": lambda x: list(x)})
            .reset_index()
        )
        gainPerSplit.loc[:, "mean"] = gainPerSplit.gains.agg(lambda x: np.mean(x))
        gainPerSplit.loc[:, "predictor"] = gainPerSplit.split.agg(
            lambda x: self.parseSplitValues(x)[0]
        )
        gainPerSplit.loc[:, "sign"] = gainPerSplit.split.agg(
            lambda x: self.parseSplitValues(x)[1]
        )
        used = gainPerSplit.apply(lambda row: self.parseSplitValues(row)[2], axis=1)

        gainPerSplit.loc[:, "values"] = used
        gainPerSplit.loc[:, "n"] = gainPerSplit.gains.str.len()
        return gainPerSplit

    def getSplitsRecursively(
        self, tree: Dict, splits: List, gains: List
    ) -> Tuple[List, List]:
        """Recursively finds splits and their gains for each node.

        By Python's mutatable list mechanic, the easiest way to achieve
        this is to explicitly supply the function with empty lists.
        Therefore, the 'splits' and 'gains' parameter expect
        empty lists when initially called.

        Parameters
        ----------
        tree: Dict
        splits: List
        gains: List

        Returns
        -------
            Tuple[List, List]
            Each split, and its corresponding gain

        """
        for key, value in tree.items():
            if key == "gain":
                if value > 0:
                    gains.append(value)
            if key == "split":
                splits.append(value)
            if key in {"left", "right"}:
                self.getSplitsRecursively(tree=dict(value), splits=splits, gains=gains)
        return splits, gains

    def plotSplitsPerVariable(self, subset: Optional[Set] = None, show=True):
        """Plots the splits for each variable in the tree.

        Parameters
        ----------
        subset: Optional[Set]
            Optional parameter to subset the variables to plot
        show: bool
            Whether to display each plot

        Returns
        -------
        plt.figure
        """
        figlist = []
        for name, plotdf in self.gainsPerSplit.groupby("predictor"):
            if (subset is not None and name in subset) or subset is None:
                fig = make_subplots()
                fig.add_trace(go.Box(x=plotdf["split"], y=plotdf["gains"], name="Gain"))
                fig.add_trace(
                    go.Scatter(
                        x=self.groupedGainsPerSplit.query(f'predictor=="{name}"')[
                            "split"
                        ],
                        y=self.groupedGainsPerSplit.query(f'predictor=="{name}"')["n"],
                        name="Number of splits",
                        mode="lines+markers",
                    )
                )
                fig.update_xaxes(
                    categoryorder="array",
                    categoryarray=self.groupedGainsPerSplit.query(
                        f'predictor=="{name}"'
                    )["split"],
                )

                fig.update_layout(
                    template="none",
                    title=f"Splits on {name}",
                    xaxis_title="Split",
                    yaxis_title="Number",
                )
                if show:
                    fig.show()  # pragma: no cover
                figlist.append(fig)
        return figlist

    def getTreeStats(self) -> pd.DataFrame:
        """Generate a dataframe with useful stats for each tree"""
        stats = pd.DataFrame(
            columns=["score", "depth", "nsplits", "gains", "meangains"]
        )
        for treeID, tree in enumerate(self.model):
            score = tree["score"]
            depths = self._depth(tree) - 1
            splits, gains = self.getSplitsRecursively(tree, splits=[], gains=[])
            nsplits = len(splits)
            meangains = mean(gains) if len(gains) > 0 else 0
            stats.loc[treeID] = [score, depths, nsplits, gains, meangains]
        return stats

    def getAllValuesPerSplit(self) -> Dict:
        """Generate a dictionary with the possible values for each split"""
        splitvalues = {}
        for name, group in self.groupedGainsPerSplit.groupby("predictor"):
            if name not in splitvalues.keys():
                splitvalues[name] = set()
            for row in group.iterrows():
                if set(group["sign"]).issubset({"in", "is"}):
                    for splitvalue in row[1]["values"]:
                        splitvalues[name] = splitvalues[name].union({splitvalue})
                else:
                    splitvalues[name].add(row[1]["values"])
        return splitvalues

    def getNodesRecursively(
        self, tree: Dict, nodelist: Dict, counter: Dict, childs: List
    ) -> Tuple[Dict, List]:
        """Recursively walks through each node, used for tree representation.

        Again, nodelist, counter and childs expects
        empty dict, dict and list parameters.

        Parameters
        ----------
        tree: Dict
        nodelist: Dict
        counter: Dict
        childs: List

        Returns
        -------
            Tuple[Dict, List]
            The dictionary of nodes and the list of child nodes

        """
        checked = False

        for key, value in tree.items():

            if key in {"left", "right"}:
                nodelist[len(counter) + 1], _ = self.getNodesRecursively(
                    value, nodelist, counter, childs
                )

            else:
                nodelist[len(counter) + 1] = {}

                if key == "score":
                    counter.append(len(counter) + 1)

                nodelist[len(counter)][key] = value

            if key == "split":
                childs[len(counter)] = {"left": 0, "right": 0}

            if not checked:
                for node, children in reversed(childs.items()):
                    if children["left"] == 0:
                        childs[node]["left"] = len(counter)
                        break
                    elif children["right"] == 0:
                        childs[node]["right"] = len(counter)
                        break
                if len(counter) > 1:
                    nodelist[len(counter)]["parent_node"] = node
                checked = True

        return nodelist, childs

    @staticmethod
    def _fillChildNodeIDs(nodeinfo: Dict, childs: Dict) -> Dict:
        """Utility function to add child info to nodes"""
        for ID, children in childs.items():
            nodeinfo[ID]["left_child"] = children["left"]
            nodeinfo[ID]["right_child"] = children["right"]
        return nodeinfo

    def getTreeRepresentation(self, tree_number: int) -> Dict:
        """Generates a more usable tree representation.

        In this tree representation, each node has an ID,
        and its attributes are the attributes,
        with parent and child nodes added as well.

        Parameters
        ----------
        tree_number: int
            The number of the tree, in order of the original json

        returns: Dict
        """

        tree = self.model[tree_number]
        nodeinfo, childs = self.getNodesRecursively(
            tree, nodelist={}, childs={}, counter=[]
        )
        tree = self._fillChildNodeIDs(nodeinfo, childs)
        # This is a very crude fix
        # It seems to add the entire tree as the last key value
        # Will work on a better fix in the future
        # For now, removing the last key seems the easiest solution
        del tree[list(tree.keys())[-1]]
        return tree

    def plotTree(
        self,
        tree_number: int,
        highlighted: Optional[Union[Dict, List]] = None,
        show=True,
    ) -> pydot.Graph:
        """Plots the chosen decision tree.

        Parameters
        ----------
        tree_number: int
            The number of the tree to visualise
        highlighted: Optional[Dict, List]
            Optional parameter to highlight nodes in green
            If a dictionary, it expects an 'x': i.e., features
            with their corresponding values.
            If a list, expects a list of node IDs for that tree.

        Returns
        -------
        pydot.Graph
        """

        if isinstance(highlighted, dict):
            highlighted = self.getVisitedNodes(tree_number, highlighted)[0]
        else:
            highlighted = highlighted or []
        nodes = self.getTreeRepresentation(tree_number)
        graph = pydot.Dot("my_graph", graph_type="graph", rankdir="BT")
        for key, node in nodes.items():
            color = "green" if key in highlighted else "white"
            label = f"Score: {node['score']}"
            if "split" in node:
                split = node["split"]
                variable, sign, values = self.parseSplitValues(split)
                if sign == "in":
                    if len(values) <= 3:
                        labelname = values
                    else:
                        totallen = len(self.allValuesPerSplit[variable])
                        labelname = (
                            f"{list(values)[0:2]+['...']} ({len(values)}/{totallen})"
                        )
                    label += f"\nSplit: {variable} in {labelname}\nGain: {node['gain']}"

                else:
                    label += f"\nSplit: {node['split']}\nGain: {node['gain']}"

                graph.add_node(
                    pydot.Node(
                        name=key,
                        label=label,
                        shape="box",
                        style="filled",
                        fillcolor=color,
                    )
                )
            else:
                graph.add_node(
                    pydot.Node(
                        name=key,
                        label=label,
                        shape="ellipse",
                        style="filled",
                        fillcolor=color,
                    )
                )
            if "parent_node" in node:
                graph.add_edge(pydot.Edge(key, node["parent_node"]))

        if show:
            display(Image(graph.create_png()))  # pragma: no cover
        return graph

    def getVisitedNodes(
        self, treeID: int, x: Dict, save_all: bool = False
    ) -> Tuple[List, float, List]:
        """Finds all visited nodes for a given tree, given an x

        Parameters
        ----------

        treeID: int
            The ID of the tree
        x: Dict
            Features to split on, with their values
        save_all: bool, default = False
            Whether to save all gains for each individual split

        Returns
        -------
        List, float, List
            The list of visited nodes,
            The score of the final leaf node,
            The gains for each split in the visited nodes

        """
        tree = self.getTreeRepresentation(treeID)
        current_node_id = 1
        leaf = False
        visited = []
        scores = []
        while not leaf:
            visited += [current_node_id]
            current_node = tree[current_node_id]
            if "split" in current_node:
                variable, type, split = self.parseSplitValues(current_node["split"])
                splitvalue = f"'{x[variable]}'" if type in {"in", "is"} else x[variable]
                if save_all:
                    scores += [{current_node["split"]: current_node["gain"]}]
                if eval(f"{splitvalue} {type} {split}"):
                    current_node_id = current_node["left_child"]
                else:
                    current_node_id = current_node["right_child"]
            else:
                leaf = True
        return visited, current_node["score"], scores

    def getAllVisitedNodes(self, x: Dict) -> pd.DataFrame:
        """Loops through each tree, and records the scoring info

        Parameters
        ----------
        x: Dict
            Features to split on, with their values

        Returns
        -------
            pd.DataFrame
        """
        forestPath = dict()
        for treeID in range(0, len(self.model)):
            forestPath[treeID] = self.getVisitedNodes(treeID, x, save_all=True)
        df = pd.DataFrame(forestPath).T.sort_values(1, ascending=False)
        df.columns = ["visited_nodes", "score", "splits"]
        return df

    def score(self, x: Dict) -> float:
        """Computes the score for a given x"""
        score = self.getAllVisitedNodes(x)["score"].sum()
        return 1 / (1 + exp(-score))

    def plotContributionPerTree(self, x: Dict, show=True):
        """Plots the contribution of each tree towards the final propensity."""
        scores = self.getAllVisitedNodes(x).sort_index()
        scores["mean"] = scores["score"].expanding().mean()
        scores["scoresum"] = scores["score"].expanding().sum()
        scores["propensity"] = scores["scoresum"].apply(lambda x: 1 / (1 + exp(-x)))
        fig = px.scatter(
            scores,
            y="score",
            template="none",
            title="Score contribution per tree, for single prediction",
            labels={"index": "Tree", "score": "Score"},
        )
        fig["data"][0]["showlegend"] = True
        fig["data"][0]["name"] = "Individual scores"
        fig.add_trace(
            go.Scatter(x=scores.index, y=scores["mean"], name="Cumulative mean")
        )
        fig.add_trace(
            go.Scatter(x=scores.index, y=scores["propensity"], name="Propensity")
        )
        fig.add_trace(
            go.Scatter(
                x=[scores.index[-1]],
                y=[scores["propensity"].iloc[-1]],
                text=[scores["propensity"].iloc[-1]],
                mode="markers+text",
                textposition="top right",
                name="Final propensity",
            )
        )
        fig.update_xaxes(zeroline=False)
        if show:
            fig.show()  # pragma: no cover
        return fig
