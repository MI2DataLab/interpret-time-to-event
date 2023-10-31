# ------------------------------------- #
# Create a simpler TCGA benchmark based 
# on the following research articles:
# https://doi.org/10.1093/bib/bbaa167
# https://doi.org/10.1093/bib/bbab354  
# ------------------------------------- #


library(OpenML)
library(mlr3proba)
library(mlr3filters)
library(caret)

load("datasets/ids.RData")

nams = c(
  "LAML", "BLCA", "LGG",  "BRCA", "COAD", "ESCA", "HNSC", "KIRC", "KIRP", 
  "LIHC", "LUAD", "LUSC", "OV", "PAAD", "SARC", "SKCM", "STAD", "UCEC"
)

mods = c("clinical", "cnv", "mirna", "mutation", "rna")

p = 5

for (nam in nams) {
  print(nam)
  
  dat_part1 <- getOMLDataSet(datset_ids[[nam]][[1]])
  dat_part2 <- getOMLDataSet(datset_ids[[nam]][[2]])
  
  dat <- cbind.data.frame(dat_part1, dat_part2)
  
  if (nam == "BRCA") {
    dat_part3 <- getOMLDataSet(datset_ids[[nam]][[3]])
    dat <- cbind.data.frame(dat, dat_part3)
  }
  temp_dataset_list = list()
  
  for (mod in mods) {
    print(mod)
    
    variables <- colnames(dat)[stringr::str_ends(colnames(dat), mod)]
    dataset <- dat[, c("time", "status", variables)]
    rownames(dataset) <- dat$bcr_patient_barcode
    
    if (sum(dataset$status) > 50) {
      set.seed(123)
      task = TaskSurv$new(nam, dataset, time = "time", event = "status")
      filter = flt("variance")
      filter$calculate(task)
      
      dataset_filtered = dataset[, names(head(filter$scores, p))]
      id_valid = createFolds(dataset$status, k=10)
      id_valid_json = jsonlite::toJSON(id_valid)
      
      write(id_valid_json, file=paste0("datasets_filtered/", tolower(nam), "_cv10split.json"))
      
      temp_dataset_list = c(temp_dataset_list, dataset_filtered)
    }
  }
  
  if (sum(dataset$status) > 50) {
    dataset_filtered = do.call(cbind, temp_dataset_list)
    dataset_filtered_full = cbind(time=dataset$time, status=dataset$status, dataset_filtered)
    write.csv(dataset_filtered_full, paste0("datasets_filtered/", tolower(nam), ".csv"), row.names=FALSE)
  }
}