#!/usr/bin/env Rscript

# compare_methods.R
# Generates presence/absence heatmaps comparing CV (3Dcount), eDNA, and UVC methods.
# Outputs to runs/analysis/figures_R/

library(dplyr)
library(tidyr)
library(vegan)
library(ggplot2)
library(lubridate)
library(pheatmap)
library(ragg)
library(purrr)
library(ggVennDiagram)
library(grid)

# --- Configuration ---
run_forlabel <- FALSE # Set to TRUE to generate the vectorized .pdf labels
output_dir <- "runs/analysis/figures_R/"
# -------------------

# --- Helper Functions ---
summarize_count <- function(df, class = c("species", "genus", "family"), value, addition = NULL, add_sp_suffix = FALSE) {
  class <- match.arg(class)
  value <- rlang::enquo(value)
  addition <- rlang::enquos(addition, .ignore_empty = "all")

  if (class == "species") {
    df %>%
      filter(!is.na(species)) %>%
      group_by(date, Species, JP_name, !!!addition) %>%
      summarise(total = sum(!!value, na.rm = TRUE), .groups = "drop")

  } else if (class == "genus") {
    res <- df %>%
      filter(!is.na(genus)) %>%
      group_by(date, genus, !!!addition) %>%
      summarise(total = sum(!!value, na.rm = TRUE), .groups = "drop") 
    if(add_sp_suffix) {
      res <- res %>% mutate(Species = paste(genus, "sp."))
    } else {
      res <- res %>% mutate(Species = genus)
    }
    res %>% select(-genus)

  } else if (class == "family") {
    res <- df %>%
      filter(!is.na(family)) %>%
      group_by(date, family, !!!addition) %>%
      summarise(total = sum(!!value, na.rm = TRUE), .groups = "drop")
    if(add_sp_suffix) {
      res <- res %>% mutate(Species = paste(family, "sp."))
    } else {
      res <- res %>% mutate(Species = family)
    }
    res %>% select(-family)
  }
}

make_table <- function(df) {
  otu_wide <- df %>%
    select(date, Species, total) %>%
    tidyr::pivot_wider(
      names_from = Species,
      values_from = total,
      values_fill = 0
    )

  summary_df <- df %>%
    group_by(date) %>%
    summarise(
      counts = sum(total, na.rm = TRUE),
      species_richness = n_distinct(Species),
      .groups = "drop"
    )

  otu_wide %>%
    left_join(summary_df, by = "date") %>%
    relocate(date, .after = last_col())
}

filter_by_time <- function(otu_df, start_time, end_time) {
  otu_df %>%
    mutate(date = as.POSIXct(date)) %>%
    filter(date >= as.POSIXct(start_time),
           date <= as.POSIXct(end_time))
}

# --- Data Loading ---
df3D <- read.csv('runs/analysis/csv_R/df3Dcount.csv', fileEncoding = "UTF-8")
dfedna <- read.csv('metadata/dfedna.csv', fileEncoding = "UTF-8")
dfuvc <- read.csv('metadata/dfuvc.csv', fileEncoding = "UTF-8")

start_time <- "2025-11-07 00:00:00"
end_time <- "2025-11-28 00:00:00"
ranges <- list(
  c(as.POSIXct("2025-11-07 00:00:00"), as.POSIXct("2025-11-08 00:00:00")),
  c(as.POSIXct("2025-11-17 00:00:00"), as.POSIXct("2025-11-18 00:00:00")),
  c(as.POSIXct("2025-11-26 00:00:00"), as.POSIXct("2025-11-27 00:00:00")))

width_t  <- 800
height_t <- 3000
colfontsize <- 18
cellheight <- 20
custom_rows2 <- c("UVC", "Video", "eDNA", "allVideo")


# ==========================================
# 1. Species Level Comparison
# ==========================================

dfuvc1 <- dfuvc %>% summarize_count(class="species", value=Count, addition=Transect, add_sp_suffix=TRUE) %>% subset(Transect==1)
dfuvc2 <- dfuvc %>% summarize_count(class="species", value=Count, addition=Transect, add_sp_suffix=TRUE) %>% subset(Transect==2)
dfuvc3 <- dfuvc %>% summarize_count(class="species", value=Count, addition=Transect, add_sp_suffix=TRUE) %>% subset(Transect==3)
df3D1  <- df3D %>% summarize_count(class="species", value=count, add_sp_suffix=TRUE) 
dfedna1<- dfedna %>% summarize_count(class="species", value=ncopiesperml, add_sp_suffix=TRUE) 

df_list_sp <- list(dfuvc1, dfuvc2, dfuvc3, df3D1, dfedna1) 
species_map <- df_list_sp %>%
  map_df(~select(.x, Species, JP_name)) %>%
  distinct(Species, .keep_all = TRUE) %>%
  filter(!is.na(Species))

df3D1sub <- df3D1 %>% mutate(date = as.POSIXct(date)) %>%
    filter(date >= as.POSIXct(start_time), date <= as.POSIXct(end_time))
species_map_cv <- list(df3D1sub) %>%
  map_df(~select(.x, Species, JP_name)) %>%
  distinct(Species, .keep_all = TRUE) %>%
  filter(!is.na(Species))

edna1 <- dfedna1 %>% make_table(.) %>% filter_by_time(start_time, end_time)
uvc1 <- dfuvc1 %>% make_table(.) %>% filter_by_time(start_time, end_time)
uvc2 <- dfuvc2 %>% make_table(.) %>% filter_by_time(start_time, end_time)
uvc3 <- dfuvc3 %>% make_table(.) %>% filter_by_time(start_time, end_time)
cv1 <- df3D1 %>% make_table(.) %>% filter_by_time(start_time, end_time)

df_list_tables <- list(uvc1 = uvc1, uvc2 = uvc2, uvc3 = uvc3, cv1 = cv1, edna1 = edna1)
dfall <- imap_dfr(df_list_tables, ~ .x %>%
                     select(-counts, -species_richness) %>%
                     mutate(dataset = .y)) %>%
  {
    num_cols <- select(., where(is.numeric)) %>% names()
    .[num_cols] <- lapply(.[num_cols], replace_na, 0)
    .
  }
dfall <- dfall %>% relocate(date, dataset, .after = last_col())

df_sub <- dfall %>%
  filter((date >= ranges[[1]][1] & date <= ranges[[1]][2]) |
         (date >= ranges[[2]][1] & date <= ranges[[2]][2]) |
         (date >= ranges[[3]][1] & date <= ranges[[3]][2]))

df_long <- df_sub %>%
  mutate(method = case_when(dataset %in% c("uvc1", "uvc2", "uvc3") ~ "UVC", TRUE ~ dataset)) %>%
  mutate(date_chr = format(date, "%Y-%m-%d %H:%M:%S"), date_fix = ymd_hms(date_chr), date_only = as.Date(date_fix)) %>%
  select(-date_chr, -date_fix, -date, -dataset)

df_long2 <- df_long %>%
  pivot_longer(cols = -c(date_only, method), names_to = "Species", values_to = "count") %>%
  mutate(count = replace_na(count, 0))

df_pa <- df_long2 %>%
  group_by(method, Species) %>%
  summarise(present = as.integer(sum(count, na.rm = TRUE) > 0), .groups = "drop")

pa_wide <- df_pa %>% pivot_wider(names_from = Species, values_from = present, values_fill = 0) %>% as.data.frame()
mat_pa <- pa_wide[,-1]
rownames(mat_pa) <- pa_wide$method
cv_species <- unique(species_map_cv$Species)
mat_pa <- mat_pa[, unique(c(cv_species, colnames(mat_pa)))]
is_in_cv <- as.numeric(colnames(mat_pa) %in% cv_species)
mat_pa_extended <- rbind(mat_pa, cv_all = is_in_cv)
mat_pa_extended <- mat_pa_extended[, colSums(mat_pa_extended, na.rm = TRUE) != 0]  

new_labels <- species_map$JP_name[match(colnames(mat_pa_extended), species_map$Species)]
labels_row_italic <- parse(text = paste0("italic(\"", rownames(t(mat_pa_extended)), "\")"))

ragg::agg_png(paste0(output_dir, "Heatmap_all.png"), width = width_t, height = height_t)
pheatmap(t(mat_pa_extended), cluster_rows = TRUE, color = c("white", "black"), border_color = NA, cellheight = cellheight, fontsize_col = colfontsize, fontsize_row = colfontsize, fontfamily = "ArialMT", legend = FALSE, labels_row = labels_row_italic, main = "Presence/Absence with CV Reference [Transposed]")
invisible(dev.off())

ragg::agg_png(paste0(output_dir, "Heatmap_all_JP.png"), width = width_t, height = height_t)
pheatmap(t(mat_pa_extended), cluster_rows = TRUE, color = c("white", "black"), labels_row = new_labels, fontfamily = "Hiragino Sans", cellheight = cellheight, fontsize_col = colfontsize, fontsize_row = colfontsize, border_color = NA, legend = FALSE, labels_col = custom_rows2, main = "Presence/Absence with CV Reference [Transposed]")
invisible(dev.off())

if(run_forlabel) {
  library(grid)
  ph <- pheatmap(t(mat_pa_extended), cluster_rows = TRUE, color = c("white", "black"), border_color = NA, cellheight = 6.7, fontsize_col = 7.6, fontsize_row = 7.6, legend = FALSE, labels_row = labels_row_italic, main = "Presence/Absence with CV Reference [Transposed]", silent = TRUE)
  g <- ph$gtable
  row_idx <- which(g$layout$name == "row_names")
  row_grob <- g$grobs[[row_idx]]
  for (i in seq_along(row_grob$children)) {
    if (inherits(row_grob$children[[i]], "text")) {
      row_grob$children[[i]]$gp$fontfamily <- "ArialMT"
      row_grob$children[[i]]$gp$fontface   <- "italic"
    }
  }
  g$grobs[[row_idx]] <- row_grob
  cairo_pdf(paste0(output_dir, "Heatmap_all_forlabel.pdf"), width = 6, height = 14)
  grid.newpage()
  grid.draw(g)
  invisible(dev.off())
}


# Plot Venn diagram
methods_use <- c("cv1", "UVC", "edna1")
mat_subset <- mat_pa_extended[methods_use, , drop = FALSE]
sets <- apply(mat_subset, 1, function(x) {
  colnames(mat_subset)[x == 1]
})
a <- ggVennDiagram(sets, label_alpha = 0, label_size = 10) +
  scale_fill_gradient(low = "transparent", high = "transparent") +
  theme_void()
ggsave(paste0(output_dir, "venn_diagram_species.pdf"), a, width = 6, height = 6, dpi = 300)

# ==========================================
# 2. Genus Level Comparison
# ==========================================

dfuvc1 <- dfuvc %>% summarize_count(class="genus", value=Count, addition=Transect, add_sp_suffix=FALSE) %>% subset(Transect==1)
dfuvc2 <- dfuvc %>% summarize_count(class="genus", value=Count, addition=Transect, add_sp_suffix=FALSE) %>% subset(Transect==2)
dfuvc3 <- dfuvc %>% summarize_count(class="genus", value=Count, addition=Transect, add_sp_suffix=FALSE) %>% subset(Transect==3)
df3D1  <- df3D %>% summarize_count(class="genus", value=count, add_sp_suffix=FALSE) 
dfedna1<- dfedna %>% summarize_count(class="genus", value=ncopiesperml, add_sp_suffix=FALSE) 

df_list_sp <- list(dfuvc1, dfuvc2, dfuvc3, df3D1, dfedna1) 
species_map <- df_list_sp %>% map_df(~select(.x, Species)) %>% distinct(Species, .keep_all = TRUE) %>% filter(!is.na(Species))

df3D1sub <- df3D1 %>% mutate(date = as.POSIXct(date)) %>% filter(date >= as.POSIXct(start_time), date <= as.POSIXct(end_time))
species_map_cv <- list(df3D1sub) %>% map_df(~select(.x, Species)) %>% distinct(Species, .keep_all = TRUE) %>% filter(!is.na(Species))

edna1 <- dfedna1 %>% make_table(.) %>% filter_by_time(start_time, end_time)
uvc1 <- dfuvc1 %>% make_table(.) %>% filter_by_time(start_time, end_time)
uvc2 <- dfuvc2 %>% make_table(.) %>% filter_by_time(start_time, end_time)
uvc3 <- dfuvc3 %>% make_table(.) %>% filter_by_time(start_time, end_time)
cv1 <- df3D1 %>% make_table(.) %>% filter_by_time(start_time, end_time)

df_list_tables <- list(uvc1 = uvc1, uvc2 = uvc2, uvc3 = uvc3, cv1 = cv1, edna1 = edna1)
dfall <- imap_dfr(df_list_tables, ~ .x %>% select(-counts, -species_richness) %>% mutate(dataset = .y)) %>%
  {
    num_cols <- select(., where(is.numeric)) %>% names()
    .[num_cols] <- lapply(.[num_cols], replace_na, 0)
    .
  }
dfall <- dfall %>% relocate(date, dataset, .after = last_col())

df_sub <- dfall %>% filter((date >= ranges[[1]][1] & date <= ranges[[1]][2]) | (date >= ranges[[2]][1] & date <= ranges[[2]][2]) | (date >= ranges[[3]][1] & date <= ranges[[3]][2]))

df_long <- df_sub %>% mutate(method = case_when(dataset %in% c("uvc1", "uvc2", "uvc3") ~ "UVC", TRUE ~ dataset)) %>%
  mutate(date_chr = format(date, "%Y-%m-%d %H:%M:%S"), date_fix = ymd_hms(date_chr), date_only = as.Date(date_fix)) %>% select(-date_chr, -date_fix, -date, -dataset)

df_long2 <- df_long %>% pivot_longer(cols = -c(date_only, method), names_to = "Species", values_to = "count") %>% mutate(count = replace_na(count, 0))

df_pa <- df_long2 %>% group_by(method, Species) %>% summarise(present = as.integer(sum(count, na.rm = TRUE) > 0), .groups = "drop")

pa_wide <- df_pa %>% pivot_wider(names_from = Species, values_from = present, values_fill = 0) %>% as.data.frame()
mat_pa <- pa_wide[,-1]
rownames(mat_pa) <- pa_wide$method
cv_species <- unique(species_map_cv$Species)
mat_pa <- mat_pa[, unique(c(cv_species, colnames(mat_pa)))]
is_in_cv <- as.numeric(colnames(mat_pa) %in% cv_species)
mat_pa_extended <- rbind(mat_pa, cv_all = is_in_cv)
mat_pa_extended <- mat_pa_extended[, colSums(mat_pa_extended, na.rm = TRUE) != 0]  

labels_row_italic <- parse(text = paste0("italic(\"", rownames(t(mat_pa_extended)), "\")"))

ragg::agg_png(paste0(output_dir, "Heatmap_genus_all.png"), width = 700, height = 3000)
pheatmap(t(mat_pa_extended), cluster_rows = TRUE, cluster_cols = TRUE, clustering_method = "average", color = c("white", "black"), border_color = NA, cellheight = 20, fontsize_col = 23, fontsize_row = 23, legend = FALSE, labels_col = custom_rows2, labels_row = labels_row_italic, main = "Presence/Absence with CV Reference [Transposed]")
invisible(dev.off())

if(run_forlabel) {
  library(grid)
  ph <-pheatmap(t(mat_pa_extended), cluster_rows = TRUE, cluster_cols = TRUE, clustering_method = "average", color = c("white", "black"), border_color = NA, cellheight = 6.7, fontsize_col = 7.6, fontsize_row = 7.6, legend = FALSE, labels_col = custom_rows2, labels_row = labels_row_italic, main = "Presence/Absence with CV Reference [Transposed]")
  g <- ph$gtable
  row_idx <- which(g$layout$name == "row_names")
  row_grob <- g$grobs[[row_idx]]
  for (i in seq_along(row_grob$children)) {
    if (inherits(row_grob$children[[i]], "text")) {
      row_grob$children[[i]]$gp$fontfamily <- "ArialMT"
      row_grob$children[[i]]$gp$fontface   <- "italic"
    }
  }
  g$grobs[[row_idx]] <- row_grob
  cairo_pdf(paste0(output_dir, "Heatmap_genus_all_forlabel.pdf"), width = 6, height = 14)
  grid.newpage()
  grid.draw(g)
  invisible(dev.off())
}


# Plot Venn diagram
methods_use <- c("cv1", "UVC", "edna1")
mat_subset <- mat_pa_extended[methods_use, , drop = FALSE]
sets <- apply(mat_subset, 1, function(x) {
  colnames(mat_subset)[x == 1]
})
a <- ggVennDiagram(sets, label_alpha = 0, label_size = 10) +
  scale_fill_gradient(low = "transparent", high = "transparent") +
  theme_void()
ggsave(paste0(output_dir, "venn_diagram_genus.pdf"), a, width = 6, height = 6, dpi = 300)

# ==========================================
# 3. Family Level Comparison
# ==========================================

dfuvc1 <- dfuvc %>% summarize_count(class="family", value=Count, addition=Transect, add_sp_suffix=FALSE) %>% subset(Transect==1)
dfuvc2 <- dfuvc %>% summarize_count(class="family", value=Count, addition=Transect, add_sp_suffix=FALSE) %>% subset(Transect==2)
dfuvc3 <- dfuvc %>% summarize_count(class="family", value=Count, addition=Transect, add_sp_suffix=FALSE) %>% subset(Transect==3)
df3D1  <- df3D %>% summarize_count(class="family", value=count, add_sp_suffix=FALSE) 
dfedna1<- dfedna %>% summarize_count(class="family", value=ncopiesperml, add_sp_suffix=FALSE) 

df_list_sp <- list(dfuvc1, dfuvc2, dfuvc3, df3D1, dfedna1) 
species_map <- df_list_sp %>% map_df(~select(.x, Species)) %>% distinct(Species, .keep_all = TRUE) %>% filter(!is.na(Species))

df3D1sub <- df3D1 %>% mutate(date = as.POSIXct(date)) %>% filter(date >= as.POSIXct(start_time), date <= as.POSIXct(end_time))
species_map_cv <- list(df3D1sub) %>% map_df(~select(.x, Species)) %>% distinct(Species, .keep_all = TRUE) %>% filter(!is.na(Species))

edna1 <- dfedna1 %>% make_table(.) %>% filter_by_time(start_time, end_time)
uvc1 <- dfuvc1 %>% make_table(.) %>% filter_by_time(start_time, end_time)
uvc2 <- dfuvc2 %>% make_table(.) %>% filter_by_time(start_time, end_time)
uvc3 <- dfuvc3 %>% make_table(.) %>% filter_by_time(start_time, end_time)
cv1 <- df3D1 %>% make_table(.) %>% filter_by_time(start_time, end_time)

df_list_tables <- list(uvc1 = uvc1, uvc2 = uvc2, uvc3 = uvc3, cv1 = cv1, edna1 = edna1)
dfall <- imap_dfr(df_list_tables, ~ .x %>% select(-counts, -species_richness) %>% mutate(dataset = .y)) %>%
  {
    num_cols <- select(., where(is.numeric)) %>% names()
    .[num_cols] <- lapply(.[num_cols], replace_na, 0)
    .
  }
dfall <- dfall %>% relocate(date, dataset, .after = last_col())

df_sub <- dfall %>% filter((date >= ranges[[1]][1] & date <= ranges[[1]][2]) | (date >= ranges[[2]][1] & date <= ranges[[2]][2]) | (date >= ranges[[3]][1] & date <= ranges[[3]][2]))

df_long <- df_sub %>% mutate(method = case_when(dataset %in% c("uvc1", "uvc2", "uvc3") ~ "UVC", TRUE ~ dataset)) %>%
  mutate(date_chr = format(date, "%Y-%m-%d %H:%M:%S"), date_fix = ymd_hms(date_chr), date_only = as.Date(date_fix)) %>% select(-date_chr, -date_fix, -date, -dataset)

df_long2 <- df_long %>% pivot_longer(cols = -c(date_only, method), names_to = "Species", values_to = "count") %>% mutate(count = replace_na(count, 0))

df_pa <- df_long2 %>% group_by(method, Species) %>% summarise(present = as.integer(sum(count, na.rm = TRUE) > 0), .groups = "drop")

pa_wide <- df_pa %>% pivot_wider(names_from = Species, values_from = present, values_fill = 0) %>% as.data.frame()
mat_pa <- pa_wide[,-1]
rownames(mat_pa) <- pa_wide$method
cv_species <- unique(species_map_cv$Species)
mat_pa <- mat_pa[, unique(c(cv_species, colnames(mat_pa)))]
is_in_cv <- as.numeric(colnames(mat_pa) %in% cv_species)
mat_pa_extended <- rbind(mat_pa, cv_all = is_in_cv)
mat_pa_extended <- mat_pa_extended[, colSums(mat_pa_extended, na.rm = TRUE) != 0]  

labels_row_italic <- parse(text = paste0("italic(\"", rownames(t(mat_pa_extended)), "\")"))

ragg::agg_png(paste0(output_dir, "Heatmap_family_all.png"), width = 700, height = 3000)
pheatmap(t(mat_pa_extended), cluster_rows = TRUE, cluster_cols = TRUE, clustering_method = "average", color = c("white", "black"), border_color = NA, cellheight = 20, fontsize_col = 18, fontsize_row = 18, legend = FALSE, labels_col = custom_rows2, main = "Presence/Absence with CV Reference [Transposed]")
invisible(dev.off())

if(run_forlabel) {
  library(grid)
  ph <-pheatmap(t(mat_pa_extended), cluster_rows = TRUE, cluster_cols = TRUE, clustering_method = "average", color = c("white", "black"), border_color = NA, cellheight = 6.7, fontsize_col = 7.6, fontsize_row = 7.6, legend = FALSE, labels_col = custom_rows2, main = "Presence/Absence with CV Reference [Transposed]")
  g <- ph$gtable
  row_idx <- which(g$layout$name == "row_names")
  row_grob <- g$grobs[[row_idx]]
  for (i in seq_along(row_grob$children)) {
    if (inherits(row_grob$children[[i]], "text")) {
      row_grob$children[[i]]$gp$fontfamily <- "ArialMT"
      row_grob$children[[i]]$gp$fontface   <- "italic"
    }
  }
  g$grobs[[row_idx]] <- row_grob
  cairo_pdf(paste0(output_dir, "Heatmap_family_all_forlabel.pdf"), width = 6, height = 14)
  grid.newpage()
  grid.draw(g)
  invisible(dev.off())
}


# Plot Venn diagram
methods_use <- c("cv1", "UVC", "edna1")
mat_subset <- mat_pa_extended[methods_use, , drop = FALSE]
sets <- apply(mat_subset, 1, function(x) {
  colnames(mat_subset)[x == 1]
})
a <- ggVennDiagram(sets, label_alpha = 0, label_size = 10) +
  scale_fill_gradient(low = "transparent", high = "transparent") +
  theme_void()
ggsave(paste0(output_dir, "venn_diagram_family.pdf"), a, width = 6, height = 6, dpi = 300)

print("Comparison plotting completed successfully.")
