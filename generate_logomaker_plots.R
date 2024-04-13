

## change current working directory
setwd("C:/Users/harol/Downloads/peter's proposal/YCFS_proposal_material/BCR_cluster/kyle-np-angela-new-approach/data_for_logoplots")


library("devtools")
install_github("omarwagih/ggseqlogo")

require(ggplot2)
require(ggseqlogo)

data <- read.csv("CL3_logoplot.csv")


# Plot a sequence logo
plot <- ggplot() + geom_logo(data["X0"], method='probability',
                             stack_width = 0.95, font='roboto_bold',
                             col_scheme='auto') + theme_logo() +
  ggtitle("Abs_induced_by_CL2")

plot

## add a title to the sequencing plot
plot + scale_y_continuous(labels=scales::percent)

plot

data <- mutate(logvalues = log10(values) - min(log10(0.0001)))
