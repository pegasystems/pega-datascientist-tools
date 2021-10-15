# Plot Control and Test groups conversion for the action/treatment combination
# NBAD uses OOTB 2% of customers for random control group
# Calculate z-score and visualize significance
# Based on Accept and Reject outcomes.
# For CTR use Clicked and Impression outcomes for Web. Set Total = Impression count,
# do not sum with Clicked outcomes


library(cdhtools)
library(data.table)
library(lubridate)
library(ggplot2)
library(tidyverse)
library(ggthemr)
library(arrow)


# Filter IH data by action id filter: may be Issue/Group/Name/Treatment
action_id_filter <- ""

#Function to calculate Z-score, p-value and confidence interval,
#as soon as R performs Chi^2 test. should be used if control group size > 30
z.prop <-
  function(x1,
           x2,
           n1,
           n2,
           conf.level = 0.95,
           alternative = "two.sided") {
    numerator <- (x1 / n1) - (x2 / n2)
    p.common <- (x1 + x2) / (n1 + n2)
    denominator <-
      sqrt(p.common * (1 - p.common) * (1 / n1 + 1 / n2))
    z.prop.ris <-
      numerator / denominator
    p1 <- x1 / n1
    p2 <- x2 / n2
    p.val <- 2 * pnorm(-abs(z.prop.ris))
    if (alternative == "lower") {
      p.val <- pnorm(z.prop.ris)
    }
    if (alternative == "greater") {
      p.val <- 1 - (pnorm(z.prop.ris))
    }
    SE.diff <- sqrt(((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2)))
    diff.prop <- (p1 - p2)
    CI <- (p1 - p2) + c(-1 * ((qnorm(((1 - conf.level) / 2
    ) + conf.level)) * SE.diff),
    ((qnorm(((1 - conf.level) / 2
    ) + conf.level)) * SE.diff))

    return (list(
      estimate = c(p1, p2),
      ts.z = z.prop.ris,
      p.val = p.val,
      diff.prop = diff.prop,
      CI = CI
    ))

  }

#Substring from the nth character
substrLeft <- function(x, n) {
  substr(x, n + 1, nchar(x))
}

#Load test data. Use read_json_arrow from arrow package to load huge datasets
load("../../r/data/ihsampledata.rda")

# Both "Accept" and "Accepted" occur as labels - fix that for the reporting
ihsampledata[Outcome == "Accept", Outcome := "Accepted"]

ih <- ihsampledata

# standardized camel casing of fields
applyUniformPegaFieldCasing(ih)

#Select Web channel outcomes and set ActionID = Adaptive model identifier if using treatment level models
#Treatment level models is a preferred approach for NBAD based CDH implementations
#Use Issue/Group/Name if using action level models
ih <- ih %>%
  filter((Outcome == "Accepted" |
            Outcome == "Rejected") & (Channel == "Web")) %>%
  mutate(ActionID = paste(Issue, "/", Group, "/", Name, "/", Treatment, sep = ""))

#Calculate acceptance rate/click-through rate
ih_analysis <- ih %>%
  group_by(ActionID, Modelcontrolgroup, Outcome) %>%
  filter(Modelcontrolgroup != "") %>%
  summarise(Count = n()) %>%
  pivot_wider(names_from = Outcome, values_from = Count) %>%
  mutate(across(everything(), ~ replace_na(.x, 0))) %>%
  mutate(Total = Accepted + Rejected,
         Conversion = 100 * Accepted / Total,
         ModelName = substrLeft(ActionID, gregexpr("/", ActionID)[[1]][-1]))

#Filter out actions
ih_action = filter(ih_analysis,
                   startsWith(ActionID, action_id_filter))

#Plot acceptance rate for test and control groups
ggthemr("flat dark")
hist_action_cmp <- ih_action %>%
  ggplot(aes(x = Modelcontrolgroup, y = Conversion)) +
  geom_col(aes(fill = Modelcontrolgroup),
           color = "darkslategray",
           alpha = 0.5) +
  facet_grid(
    ~ gsub("/", "\n", ModelName),
    scales = "free_x",
    space = "free_x",
    switch = "x"
  ) +
  ggtitle("Control group conversion analysis") +
  labs(y = "Accept Rate %", x = "Action/Treatment model", fill = "Group")

hist_action_cmp

#Calculate Z-score and p-value
z_ih_action <- ih_action %>%
  pivot_wider(
    names_from = Modelcontrolgroup,
    values_from = c(Accepted, Rejected, Total, Conversion)
  ) %>%
  mutate(across(everything(), ~ replace_na(.x, 0))) %>%
  mutate(ZScore = (
    z.prop(Accepted_Control, Accepted_Test, Total_Control, Total_Test)$ts.z
  ))

#Plot Z-score
z_test <- z_ih_action %>%
  ggplot(aes(x = gsub("/", "\n", ModelName), y = ZScore)) +
  geom_col() +
  ggtitle("Z-score") +
  labs(y = "Z", x = "Action/Treatment")

z_test

#Save a table of all models sorted by p-value. Useful to evaluate lift
evaluate_models <- ih_action %>%
  pivot_wider(
    names_from = Modelcontrolgroup,
    values_from = c(Accepted, Rejected, Total, Conversion)
  ) %>%
  mutate(across(everything(), ~ replace_na(.x, 0))) %>%
  mutate(ZScore = (
    z.prop(Accepted_Control, Accepted_Test, Total_Control, Total_Test)$ts.z
  ),
  P.value = (
    z.prop(Accepted_Control, Accepted_Test, Total_Control, Total_Test)$p.val
  )) %>%
  arrange(P.value)

write_excel_csv(evaluate_models, "~/ModelsWithStatSignificantDiffFromRandom.csv")
