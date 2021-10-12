# Plot Control and Test groups CTR for the action/treatment combination
# Calculate z-score and visualize significance

library(cdhtools)
library(data.table)
library(lubridate)
library(ggplot2)
library(tidyverse)
library(ggthemr)


# Filter IH dasta by action id filter: may be Issue/Group/Name/Treatment
action_id_filter <- "Awareness/Pro"

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

substrLeft <- function(x, n) {
  substr(x, n + 1, nchar(x))
}

ih <-
  readDSExport(
    "Data-pxStrategyResult_InteractionHistorySample_20211011T175050_GMT.zip",
    "~/Downloads"
  )

# standardized camel casing of fields
applyUniformPegaFieldCasing(ih)

# convert date/time fields
ih[, OutcomeTime := fromPRPCDateTime(OutcomeTime)]
ih <- ih %>%
  mutate(OutcomeDateTime = fromPRPCDateTime(OutcomeTime)) %>%
  arrange(desc(OutcomeDateTime)) %>%
  filter((Outcome == "Clicked" |
            Outcome == "Impression") & (Channel == "Web")) %>%
  mutate(ActionID = paste(Issue, "/", Group, "/", Name, "/", Treatment, sep = ""))

ih_analysis <- ih %>%
  group_by(ActionID, Modelcontrolgroup, Outcome) %>%
  filter(Modelcontrolgroup != "") %>%
  summarise(Count = n()) %>%
  pivot_wider(names_from = Outcome, values_from = Count) %>%
  mutate(across(everything(), ~ replace_na(.x, 0))) %>%
  mutate(Total = Impression + Clicked, CTR = 100 * Clicked / Total) %>%
  mutate(ModelName = substrLeft(ActionID, gregexpr("/", ActionID)[[1]][-1]))

ih_action = filter(ih_analysis,
                   startsWith(ActionID, action_id_filter) & !endsWith(ActionID, "/"))

ggthemr("flat dark")
hist_action_cmp <- ih_action %>%
  ggplot(aes(x = Modelcontrolgroup, y = CTR)) +
  geom_col(aes(fill = Modelcontrolgroup),
           color = "darkslategray",
           alpha = 0.5) +
  facet_grid(
    ~ gsub("/", "\n", ModelName),
    scales = "free_x",
    space = "free_x",
    switch = "x"
  ) +
  ggtitle("Control group CTR analysis") +
  labs(y = "Click Through Rate %", x = "Action/Treatment model") +
  scale_y_continuous(breaks = seq(0, 5, 0.2))

hist_action_cmp

z_ih_action <- ih_action %>%
  pivot_wider(names_from = Modelcontrolgroup,
              values_from = c(Impression, Clicked, Total, CTR)) %>%
  mutate(across(everything(), ~ replace_na(.x, 0))) %>%
  mutate(ZScore = (
    z.prop(Clicked_Control, Clicked_Test, Total_Control, Total_Test)$ts.z
  ))


z_test <- z_ih_action %>%
  ggplot(aes(x = gsub("/", "\n", ModelName), y = ZScore)) +
  geom_col() +
  ggtitle("Z-score visualisation") +
  labs(y = "Z", x = "Action/Treatment")

z_test

good_treatment_models <-
  filter(z_ih_action, abs(ZScore) > 1.96 & !endsWith(ActionID, "/"))
