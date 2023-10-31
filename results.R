# ------------------------------------- #
# Visualize results of 4.2
# ------------------------------------- #


library(ggplot2)
library(tidyr)
library(dplyr) 
library(patchwork)
library(stringr)

datasets = c("blca", "lgg", "brca", "hnsc", "kirc", "luad", 
             "lusc", "ov", "paad", "skcm", "stad")


#### EDA: time

df_list = list()
for (d in datasets) {
  df = read.csv(paste0("datasets/", d, ".csv"))
  df$dataset = d
  df_list[[d]] = df[, c("time", "dataset")]
}
df_meta = do.call(rbind, df_list)

ggplot(df_meta %>% 
         group_by(dataset) %>% 
         mutate(time_point=quantile(time, 0.95)) %>%
         ungroup()) + 
  geom_histogram(aes(x=time)) + 
  geom_vline(aes(xintercept=time_point), color="red") +
  scale_x_log10() +
  facet_wrap(~dataset) +
  ggtitle("time distribution with 95% quantile")

time_max = df_meta %>% 
  group_by(dataset) %>% 
  summarise(time_point=quantile(time, 0.75)) %>%
  tibble::deframe()
time_max        


#### GPFI in CV

performance = read.csv("outputs/performance.csv", 
                       header=FALSE,
                       col.names = c("model", "experiment", "ibs"))
performance_cv = performance %>%
  filter(stringr::str_detect(experiment, "split")) %>%
  mutate(split = stringr::str_extract(experiment, "\\d+$"),
         dataset = stringr::str_split(experiment, "_", simplify = TRUE)[,1]) %>%
  group_by(dataset) %>%
  summarise(ibs_mean = mean(ibs), ibs_sd = sd(ibs))

results_list = list()
for (d in datasets) {
  temp = read.csv(paste0("outputs/gpfi_", d, "_split=1.csv"))[,-1]
  temp$split = 1
  for (split in 2:10) {
    temp2 = read.csv(paste0("outputs/gpfi_", d, "_split=", split, ".csv"))[,-1]  
    temp2$split = split
    temp = rbind(temp, temp2)
  }
  results_list[[d]] = temp %>% 
    pivot_longer(clinical:rna, names_to = "modality") %>%
    mutate(value = value - full_model) %>%
    select(-full_model) %>%
    spread(modality, value) %>%
    group_by(timesteps) %>%
    summarise_at(vars(clinical:rna), mean) %>% # sd
    mutate(dataset = d) 
}

results_df = do.call(rbind, results_list) %>%
  pivot_longer(clinical:rna, names_to = "modality")

performance_dict_mean = as.list(performance_cv$ibs_mean)
names(performance_dict_mean) = performance_cv$dataset
performance_dict_sd = as.list(performance_cv$ibs_sd)
names(performance_dict_sd) = performance_cv$dataset

results_df$dataset_nice = paste0(
  toupper(results_df$dataset), 
  " (", round(unlist(performance_dict_mean[results_df$dataset]), 2),
  "+-", round(unlist(performance_dict_sd[results_df$dataset]), 2),
  ")"
)

p1 = ggplot(results_df %>% 
              group_by(dataset) %>% 
              filter(timesteps <= time_max[dataset]) %>% 
              ungroup()) +
  # geom_smooth(aes(x=timesteps, y=value, color=modality), se=FALSE) +
  geom_line(aes(x=timesteps, y=value, color=modality), linewidth=1) +
  facet_wrap(~dataset_nice, scales="free") +
  labs(y="Increase in brier score after permutation",
       x="Days",
       color="Modality",
       title="Grouped permutation feature importance for 11 multi-omics datasets from TCGA") +
  scale_color_discrete(type=DALEX::colors_discrete_drwhy(5)) +
  DALEX::theme_drwhy()
p1
ggsave(filename = "figures/tcga_gpfi_cv.png", plot = p1, width = 9, height=6, bg="white")


#### PDP with SD

performance_pdp = tail(read.csv("outputs/performance_pdp.csv", 
                       header=FALSE,
                       col.names = c("model", "experiment", "ibs")), 11)
performance_pdp
performance_dict = as.list(performance_pdp$ibs)
datasets = c("blca", "lgg", "brca", "hnsc", "kirc", "luad", 
             "lusc", "ov", "paad", "skcm", "stad")
names(performance_dict) = datasets
performance_dict

plot_pdp <- function(dataset) {
  pdp_mean_raw = read.csv(paste0("outputs/pdp_diff-var=false_full-train=true_", dataset ,".csv"))[,-1]
  pdp_var_raw = read.csv(paste0("outputs/pdp_diff-var=true_full-train=true_", dataset ,".csv"))[,-1]
  
  pdp_sd_raw = sqrt(pdp_var_raw[, -c(1, 2)])
  pdp_sd = cbind(pdp_var_raw$timesteps, pdp_sd_raw)
  
  var_raw = str_remove(colnames(pdp_sd_raw), "_clinical")
  value_raw = str_replace_all(str_split_i(var_raw, "_", -1), "\\.", " ")
  var_nice = paste0(
    str_to_sentence(str_replace_all(str_extract(var_raw, "^[^A-Z]*"), "_", " ")),
    "= ",
    ifelse(tolower(value_raw) == value_raw, "Yes", value_raw)
  )
  
  
  # manual edits
  var_nice = stringr::str_replace(
    stringr::str_replace(var_nice, "history=", "history ="),
    "Stage event p", "P")
  #
  
  colnames(pdp_sd) = c("timesteps", var_nice)
  
  pdp_mean = pdp_mean_raw[, endsWith(colnames(pdp_mean_raw), "1")] - pdp_mean_raw[, endsWith(colnames(pdp_mean_raw), "0")]
  pdp_mean = cbind(pdp_mean_raw$timesteps, pdp_mean)
  colnames(pdp_mean) = colnames(pdp_sd)
  
  pdp_df = cbind(
    pdp_mean %>%
      pivot_longer(!timesteps, names_to = "feature", values_to = "mean"),
    pdp_sd %>%
      pivot_longer(!timesteps, names_to = "feature", values_to = "sd") %>%
      select(sd)
  )
  
  p = ggplot(pdp_df %>% filter(timesteps <= time_max[dataset])) +
    geom_line(aes(x=timesteps, y=mean, color=feature), linewidth=1) +
    geom_ribbon(aes(x=timesteps, ymin=mean-sd, ymax=mean+sd, fill=feature),
                alpha=0.25) +
    labs(y="Relative feature effect",
         x="Days",
         color=NULL,
         title=paste("Partial dependence of clinical features in", toupper(dataset))
    ) +
    scale_color_discrete(type=DALEX::colors_discrete_drwhy(4)) +
    scale_fill_discrete(guide=NULL, type=DALEX::colors_discrete_drwhy(4)) +
    DALEX::theme_drwhy() +
    guides(color=guide_legend(ncol=2))
  p
}

for (d in datasets) {
  p = plot_pdp(d) 
  ggsave(filename = paste0("figures/tcga_pdp_", d, "_clinical.png"),
         plot = p, width = 6, height=3.5, bg="white")
}


