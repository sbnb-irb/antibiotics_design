library(ape)
library(tidyverse)
library(castor)

file <- "input_file_gtdb_taxonomy.tsv"

tax <- read_delim(file,
                  delim = "\t", show_col_types = F) %>% 
  separate(lineage, c("k", "p", "c", "o", "f", "g", "s"), ";")  %>% 
  mutate(across(k:s, ~stringr::str_replace_all(., "[()]", ""))) %>%
  mutate_all(as.factor)

frm <- ~k/p/c/o/f/g/id
tr <- as.phylo(frm, data = tax, collapse = TRUE)

sps <- tr$tip.label

# there are many methods to compute distance among tips, the decision depends on your objective

distmat <- matrix(nrow = length(sps), ncol = length(sps))
dimnames(distmat) <- list(sps, sps)

for (species in sps) {
  distmat[species, ] <- castor::get_pairwise_distances(tr, rep(species, length(sps)), sps)
}

heatmap(distmat)

dim(distmat)

# Save results
write.csv(distmat, file = "distmat_GTDB.csv", row.names = TRUE)

# histogram
dvals <- distmat[upper.tri(distmat, diag = FALSE)]
hist(dvals, main = "Distribution of Pairwise Distances",
     xlab = "Distance", ylab = "Frequency")

library(ggplot2)

ggplot(data.frame(distance = dvals), aes(x = distance)) +
  geom_histogram(bins = 50) +
  labs(title = "Distribution of Pairwise Distances",
       x = "Distance", y = "Count")
