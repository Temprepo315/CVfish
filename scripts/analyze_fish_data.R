#!/usr/bin/env Rscript

# analyze_fish_data.R
# Outputs generated CSVs and figures to runs/analysis/csv_R/ and runs/analysis/figures_R/

library(dplyr)
library(ggplot2)
library(lubridate)
library(readr)
library(tidyr)
library(stringr)

# --- 1. Data Loading and Filtering ---
df3D <- read.csv("runs/analysis/csv_3d/summary_all.csv")
df3D$date <- as.POSIXct(df3D$date, format="%Y-%m-%d %H:%M:%S")

# Remove empty names
df3D <- df3D %>% 
    filter(!is.na(refined_name), trimws(refined_name) != "")

# Stereo matching quality filters
percentage <- 0.2
df3Dsub <- df3D %>%
  filter(
    abs(heady_rect - D_heady_rect) <= percentage * h,
    abs(taily_rect - D_taily_rect) <= percentage * h
  )

# Edge constraints
df3Dsub <- df3Dsub %>%
  filter(
    D_x1 > 1,
    D_x1 < 3839,
    D_x2 > 1,
    D_x2 < 3839,
    x1 > 1,
    x1 < 3839,
    x2 > 1,
    x2 < 3839
  )

# Orientation constraints
df3Dsub <- df3Dsub %>%
  filter(angle > 135 | angle < 45)

# Extract period
period <- c(as.POSIXct("2025-11-07 08:00:00", tz = "Asia/Tokyo"),
            as.POSIXct("2025-11-27 17:00:00", tz = "Asia/Tokyo"))
df3Dsub <- df3Dsub %>% subset(date >= period[1] & date <= period[2])

# --- 2. Y-distance and Length computation ---
# For Y for all individuals at one timepoint
df3Dsub_Y <- df3Dsub %>%
  group_by(date) %>%
  summarise(Y_mean = mean(Y_xy), Y_median = median(Y_xy), Y_75 = quantile(Y_xy, probs = 0.75), .groups = "drop")

# For length for each individual
df3Dsub_length <- df3Dsub %>%
  group_by(refined_name, id, date) %>%
  summarise(TotalLength = 100 * quantile(length, probs = 0.75), .groups = "drop") ##in cm

# --- 3. Label matching and Mass Estimation ---
dflabel = read.csv("metadata/class_summary_with_JP_labeled.csv", encoding="UTF-8")
dflabel[dflabel == ""] <- NA
dflabel$Species = paste(dflabel$genus, dflabel$species, sep=" ")

dflabel <- dflabel %>%
  mutate(across(c(family, genus, species), ~ na_if(trimws(.x), ""))) %>%
  mutate(
    Species = case_when(
      !is.na(species) ~ Species,
      is.na(species) & !is.na(genus)  ~ name,
      is.na(species) & is.na(genus) & !is.na(family) ~ name,
      TRUE ~ NA_character_))
dflabel[is.na(dflabel$Species), ]$Species <- "unID"

# Pick species-level labels only
dflabel_sponly <- dflabel %>% subset(cls==1)
dflabel_sponly$Taxon <- paste(dflabel_sponly$genus, dflabel_sponly$species, sep=' ')

# Few mappings to avoid NA when combining with length-weight equations
dflabel_sponly2 <- dflabel_sponly %>%
  mutate(Taxon = case_when(
    Taxon == "Plectroglyphidodon altus" ~ "Plectroglyphidodon species", 
    Taxon == "Meiacanthus kamoharai" ~ "Meiacanthus species", 
    Taxon == "Rhinecanthus verrucosus" ~ "Rhinecanthus aculeatus", 
    TRUE ~ Taxon  
  ))

# Load length-weight equations and do the matching
dflength = read.csv("metadata/fish_lengthmass_list.csv")
dfspecies <- left_join(dflabel_sponly2, dflength, by = "Taxon")

# Mapping the species names back
dfspecies <- dfspecies %>%
  mutate(Taxon = case_when(
    Taxon == "Plectroglyphidodon species" ~ "Plectroglyphidodon altus",
    Taxon == "Meiacanthus species" ~ "Meiacanthus kamoharai" , 
    Taxon ==  "Rhinecanthus aculeatus" ~ "Rhinecanthus verrucosus", 
    TRUE ~ Taxon 
  ))

# Get weight of each individual by combining with df3Dsub_length
df3Dsub_lengthmass <- df3Dsub_length %>%
  left_join(dfspecies %>% select(name, a, b, Source.Length.to.Total.Length.Ratio, Source.Maximum.Length..cm.), by = c("refined_name"="name")) %>%
  mutate(weight = a * ((Source.Length.to.Total.Length.Ratio*TotalLength) ^ b))

# Some namings
label_to_name <- setNames(dflabel$label, dflabel$name)
label_to_JPname <- setNames(dflabel$JP_name, dflabel$name)
label_to_Species <- setNames(dflabel$Species, dflabel$name)

sorted_labels <- sort(unique(df3Dsub_length$refined_name))
sorted_names <- sapply(sorted_labels, function(lbl) {
  if (!is.null(label_to_name[[lbl]])) label_to_name[[lbl]] else lbl
})
sorted_JPnames <- sapply(sorted_labels, function(lbl) {
  if (!is.null(label_to_JPname[[lbl]])) label_to_JPname[[lbl]] else lbl
})
sorted_Species <- sapply(sorted_labels, function(lbl) {
  if (!is.null(label_to_Species[[lbl]])) label_to_Species[[lbl]] else lbl
})

df3Dsub_lengthmass$label <- sorted_names[df3Dsub_lengthmass$refined_name]
df3Dsub_lengthmass$label <- factor(df3Dsub_lengthmass$label, levels = sorted_names)

df3Dsub_lengthmass$JP_name <- sorted_JPnames[df3Dsub_lengthmass$refined_name]
df3Dsub_lengthmass$JP_name <- factor(df3Dsub_lengthmass$JP_name, levels = sorted_JPnames)

df3Dsub_lengthmass$Species <- sorted_Species[df3Dsub_lengthmass$refined_name]
df3Dsub_lengthmass$Species <- factor(df3Dsub_lengthmass$Species, levels = sorted_Species)

# --- 4. Plotting Length and Mass ---
# Total Length (m)
a_length <- ggplot(df3Dsub_lengthmass[!is.na(df3Dsub_lengthmass$a), ], aes(x = TotalLength/100, y = Species)) + 
  geom_jitter(width = 0, height = 0.3, alpha = 0.1)+
  geom_boxplot(fill = NA, color = "black", outlier.shape = NA, width=0.8) +
  geom_vline(xintercept = seq(0, 6, by = 0.1), color = "gray", linetype = "dashed", linewidth = 0.3) +
  coord_cartesian(xlim = c(0, 2.5)) +
  labs(x = "Total length", y = "species") +
  theme_bw() +
  theme(
    axis.text.y = element_text(size=20, face = "italic"),
    axis.text.x = element_text(size=20),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    text = element_text(family = "Arial")
  )
ggsave("runs/analysis/figures_R/bdsizesp_2025.png", plot = a_length, width = 15, height = 30, units = "in", dpi = 300)

# Body mass (kg)
b_mass <- ggplot(df3Dsub_lengthmass[!is.na(df3Dsub_lengthmass$a), ], aes(x = weight/1000, y = Species)) + 
  geom_jitter(width = 0, height = 0.3, alpha = 0.1)+
  geom_boxplot(fill = NA, color = "black", outlier.shape = NA, width=0.8) +
  coord_cartesian(xlim = c(0,5)) + 
  labs(x = "mass", y = "name") +
  theme_bw() +
  theme(
    axis.text.y = element_text(size=20, face = "italic"),
    axis.text.x = element_text(size=20),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    text = element_text(family = "Arial")
  )
ggsave("runs/analysis/figures_R/weightsp_2025.png", plot = b_mass, width = 15, height = 30, units = "in", dpi = 300)

# --- 5. Total Biomass and Time Series Preparation ---
dfmass_summary <- df3Dsub_lengthmass %>%
  filter(!is.na(weight), !is.na(a), !is.na(b)) %>%
  mutate(calc_weight = if_else(
    TotalLength >= 1.5 * Source.Maximum.Length..cm.,
    a * ((Source.Length.to.Total.Length.Ratio * Source.Maximum.Length..cm.) ^ b),
    weight)) %>%
  group_by(date) %>%
  summarize(sumweight = sum(calc_weight, na.rm = TRUE), .groups = "drop")

write.csv(dfmass_summary, "runs/analysis/csv_R/dfmass_summary.csv", row.names = FALSE, fileEncoding = "UTF-8")

summarize_count <- function(df, class = c("species", "genus", "family"), value, addition = NULL) {
  class <- match.arg(class)
  value <- rlang::enquo(value)
  addition <- rlang::enquos(addition, .ignore_empty = "all")

  if (class == "species") {
    df %>% filter(!is.na(species)) %>% group_by(date, Species, JP_name, !!!addition) %>%
      summarise(total = sum(!!value, na.rm = TRUE), .groups = "drop")
  } else if (class == "genus") {
    df %>% filter(!is.na(genus)) %>% group_by(date, genus, !!!addition) %>%
      summarise(total = sum(!!value, na.rm = TRUE), .groups = "drop") %>%
      mutate(Species = paste(genus, "sp.")) %>% select(-genus)
  } else if (class == "family") {
    df %>% filter(!is.na(family)) %>% group_by(date, family, !!!addition) %>%
      summarise(total = sum(!!value, na.rm = TRUE), .groups = "drop") %>%
      mutate(Species = paste(family, "sp.")) %>% select(-family)
  }
}

make_table <- function(df) {
  otu_wide <- df %>% select(date, Species, total) %>% tidyr::pivot_wider(names_from = Species, values_from = total, values_fill = 0)
  summary_df <- df %>% group_by(date) %>%
    summarise(counts = sum(total, na.rm = TRUE), species_richness = n_distinct(Species), .groups = "drop")
  otu_wide %>% left_join(summary_df, by = "date") %>% relocate(date, .after = last_col())
}

df3Dcount <- df3D %>% group_by(date, refined_name) %>% summarise(count = n_distinct(refined_id), .groups = "drop") %>% ungroup()
df3Dcount$date <- as.POSIXct(df3Dcount$date, format="%Y-%m-%d %H:%M:%S")

df3Dcount <- df3Dcount %>% left_join(dflabel, by = c("refined_name" = "name")) %>%
  mutate(across(where(is.character), ~ na_if(trimws(.x), ""))) %>%
  select(-color, -n_images, -n_instances, -num, -label)

df3Dcount2 <- df3Dcount %>% select(-cls, -sp_grp) %>%
  mutate(across(c(family, genus, species), ~ na_if(trimws(.x), ""))) %>%
  mutate(Species = case_when(
      !is.na(species) ~ Species,
      is.na(species) & !is.na(genus)  ~ paste("Genus", genus),
      is.na(species) & is.na(genus) & !is.na(family) ~ paste("Family", family),
      TRUE ~ NA_character_)) %>% filter(!is.na(Species))
write.csv(df3Dcount2, "runs/analysis/csv_R/df3Dcount.csv", row.names = FALSE, fileEncoding = "UTF-8")

library(purrr)
df3Dcount <- df3Dcount %>%
  group_split(date) %>%
  map_dfr(function(df_date) {
        df_updated <- df_date %>% subset(cls==1) %>% group_by(sp_grp) %>%
          mutate(two_total = ifelse(is.na(sp_grp), NA, sum(count)), two_ratio = ifelse(is.na(sp_grp), NA, count / sum(count))) %>% ungroup()
        cls2 <- df_date %>% subset(cls==2) %>% mutate(count2 = count)
        df_updated <- df_updated %>% left_join(cls2 %>% select(sp_grp, count2), by = "sp_grp") %>% mutate(count_up2= rowSums(cbind(count,count2*two_ratio), na.rm = TRUE))
        df_updated <- df_updated %>% subset(cls==1) %>% group_by(family, genus) %>% mutate(genus_total = sum(count), genus_ratio = count / genus_total) %>% ungroup() 
        cls4 <- df_date %>% subset(cls==4) %>% mutate(count4 = count)
        df_updated <- df_updated %>% left_join(cls4 %>% select(genus, count4), by = "genus") %>% mutate(count_up4= rowSums(cbind(count_up2,count4*genus_ratio), na.rm = TRUE))
        df_updated <- df_updated %>% subset(cls==1) %>% group_by(family) %>% mutate(family_total = sum(count), family_ratio = count / family_total) %>% ungroup() 
        cls5 <- df_date %>% subset(cls==5) %>% mutate(count5 = count)
        df_updated <- df_updated %>% left_join(cls5 %>% select(family, count5), by = "family") %>% mutate(count_new= rowSums(cbind(count_up4,count5*family_ratio), na.rm = TRUE))
    return(df_updated)
  })

df3Dcount <- df3Dcount %>% select(date, JP_name, family, genus, species, Species, count_new)
ranges <- list(c(as.POSIXct("2025-11-07 00:00:00"), as.POSIXct("2025-11-28 00:00:00")))
df3D_sub <- df3Dcount %>% filter((date >= ranges[[1]][1] & date <= ranges[[1]][2]))

summary_tbl <- df3D_sub %>% summarize_count(class="species", value=count_new) %>% make_table(.)
dfmass_summary <- read.csv("runs/analysis/csv_R/dfmass_summary.csv")
dfmass_summary$date <- as.POSIXct(dfmass_summary$date)
summary_tbl <- summary_tbl %>% left_join(dfmass_summary, by="date") %>% rename(weight=sumweight)
summary_tbl <- summary_tbl %>% left_join(df3Dsub_Y, by="date")

full_time_day <- seq(from = summary_tbl$date[1], to = summary_tbl$date[nrow(summary_tbl)], by = "1 hour") %>% .[hour(.) >= 8 & hour(.) <= 17]
summary_tbl <- tibble(date = full_time_day) %>% left_join(summary_tbl, by = "date") %>% mutate(across(-date, ~ replace_na(.x, 0)))

dfenv <- read.csv("metadata/dfturb.csv")
dfenv$date <- as.POSIXct(dfenv$date, tz = "Asia/Tokyo")
full_time <- seq(from = as.POSIXct("2025-11-07 12:00:00", tz = "Asia/Tokyo"), to = as.POSIXct("2025-11-27 10:00:00", tz = "Asia/Tokyo"), by = "1 hour")
full_time_day_env <- full_time[hour(full_time) >= 8 & hour(full_time) <= 17]
dfenv2 <- tibble(date = full_time_day_env) %>% left_join(dfenv, by = "date") 
summary_tbl <- summary_tbl %>% left_join(dfenv2, by="date")
write.csv(summary_tbl, "runs/analysis/csv_R/sp_table.csv", row.names = FALSE, fileEncoding = "UTF-8")

dfallsub_long <- summary_tbl %>% pivot_longer(cols = -date, names_to = "Species", values_to = "Value")
complete_dates <- seq(from = as.POSIXct("2025-11-07 12:00:00"), to = as.POSIXct("2025-11-27 10:00:00"), by = "hour")
complete_df <- expand.grid(date = complete_dates, Species = unique(dfallsub_long$Species))
final_df <- complete_df %>% left_join(dfallsub_long, by = c("date", "Species")) %>% mutate(Value = ifelse(is.na(Value), NA, Value))

# --- 6. Time Series Plots ---
axistitle=24
axistext=40
pointsize=5
day_start <- as.POSIXct("2000-01-01 08:00:00", tz = "Asia/Tokyo")
day_end   <- as.POSIXct("2000-01-01 17:00:00", tz = "Asia/Tokyo")

# Species Richness
finalsub_a <- final_df %>% subset(Species == "species_richness") %>% mutate(date = with_tz(date, tzone = "Asia/Tokyo"), day_label = format(date, "%m/%d"), hour_time = update(date, year = 2000, month = 1, mday = 1))
ts_richness <- ggplot(finalsub_a, aes(x = hour_time, y = Value, group = Species)) +
  geom_line() + geom_point(size = pointsize) +
  facet_grid(. ~ day_label, scales = "free_x", space = "free_x", switch = "x") + 
  scale_x_datetime(date_labels = "%H:%M", breaks = "4 hours", limits = c(day_start, day_end)) +
  theme_classic() +
  theme(panel.spacing = unit(1, "lines"), strip.background = element_blank(), strip.placement = "outside", axis.ticks = element_line(linewidth = 2), axis.ticks.length = unit(0.3, "cm"), axis.title = element_text(size = axistitle), axis.text.x = element_text(size = 30, angle = 45, hjust = 1), axis.text.y = element_text(size = axistext), strip.text = element_text(size = axistext)) +
  scale_y_continuous(labels = function(x) sprintf("%4d", x))
ggsave("runs/analysis/figures_R/species_richness.pdf", plot = ts_richness, width = 49, height = 12, limitsize=FALSE)

# Counts
finalsub_b <- final_df %>% subset(Species == "counts") %>% mutate(date = with_tz(date, tzone = "Asia/Tokyo"), day_label = format(date, "%m/%d"), hour_time = update(date, year = 2000, month = 1, mday = 1))
ts_counts <- ggplot(finalsub_b, aes(x = hour_time, y = Value, group = Species)) +
  geom_line() + geom_point(size = pointsize) +
  facet_grid(. ~ day_label, scales = "free_x", space = "free_x", switch = "x") + 
  scale_x_datetime(date_labels = "%H:%M", breaks = "4 hours", limits = c(day_start, day_end)) +
  theme_classic() +
  theme(panel.spacing = unit(1, "lines"), strip.background = element_blank(), strip.placement = "outside", axis.ticks = element_line(linewidth = 2), axis.ticks.length = unit(0.3, "cm"), axis.title = element_text(size = axistitle), axis.text.x = element_text(size = 30, angle = 45, hjust = 1), axis.text.y = element_text(size = axistext), strip.text = element_text(size = axistext)) +
  scale_y_continuous(labels = scales::label_number(accuracy = 1))
ggsave("runs/analysis/figures_R/counts.pdf", plot = ts_counts, width = 49, height = 12, limitsize=FALSE)

# Weight (kg)
finalsub_c <- final_df %>% subset(Species == "weight") %>% mutate(date = with_tz(date, tzone = "Asia/Tokyo"), day_label = format(date, "%m/%d"), hour_time = update(date, year = 2000, month = 1, mday = 1))
ts_weight <- ggplot(finalsub_c, aes(x = hour_time, y = Value/1000, group = Species)) +
  geom_line() + geom_point(size = pointsize) +
  facet_grid(. ~ day_label, scales = "free_x", space = "free_x", switch = "x") + 
  scale_x_datetime(date_labels = "%H:%M", breaks = "4 hours", limits = c(day_start, day_end)) +
  theme_classic() +
  theme(panel.spacing = unit(1, "lines"), strip.background = element_blank(), strip.placement = "outside", axis.ticks = element_line(linewidth = 2), axis.ticks.length = unit(0.3, "cm"), axis.title = element_text(size = axistitle), axis.text.x = element_text(size = 30, angle = 45, hjust = 1), axis.text.y = element_text(size = axistext), strip.text = element_text(size = axistext)) +
  scale_y_continuous(labels = function(x) sprintf("%4d", x))
ggsave("runs/analysis/figures_R/weight.pdf", plot = ts_weight, width = 49, height = 12, limitsize=FALSE)

# Turbidity (FTU)
finalsub_turb <- final_df %>% subset(Species == "turb_ftu") %>% mutate(date = with_tz(date, tzone = "Asia/Tokyo"), day_label = format(date, "%m/%d"), hour_time = update(date, year = 2000, month = 1, mday = 1))
ts_turb <- ggplot(finalsub_turb, aes(x = hour_time, y = Value, group = Species)) +
  geom_line() + geom_point(size = pointsize) +
  facet_grid(. ~ day_label, scales = "free_x", space = "free_x", switch = "x") + 
  scale_x_datetime(date_labels = "%H:%M", breaks = "4 hours", limits = c(day_start, day_end)) +
  theme_classic() +
  theme(panel.spacing = unit(1, "lines"), strip.background = element_blank(), strip.placement = "outside", axis.title = element_text(size = axistitle), axis.text.x = element_text(size = 30, angle = 45, hjust = 1), axis.text.y = element_text(size = axistext), strip.text = element_text(size = axistext)) +
  scale_y_continuous(labels = function(x) sprintf("%4d", x))
ggsave("runs/analysis/figures_R/turbidity.pdf", plot = ts_turb, width = 49, height = 12, limitsize=FALSE)

# Mean Y (m)
finalsub_ymean <- final_df %>% subset(Species == "Y_mean") %>% mutate(date = with_tz(date, tzone = "Asia/Tokyo"), day_label = format(date, "%m/%d"), hour_time = update(date, year = 2000, month = 1, mday = 1))
ts_ymean <- ggplot(finalsub_ymean, aes(x = hour_time, y = Value, group = Species)) +
  geom_line() + geom_point(size = pointsize) +
  facet_grid(. ~ day_label, scales = "free_x", space = "free_x", switch = "x") + 
  scale_x_datetime(date_labels = "%H:%M", breaks = "4 hours", limits = c(day_start, day_end)) +
  theme_classic() +
  theme(panel.spacing = unit(1, "lines"), strip.background = element_blank(), strip.placement = "outside", axis.title = element_text(size = axistitle), axis.text.x = element_text(size = 30, angle = 45, hjust = 1), axis.text.y = element_text(size = axistext), strip.text = element_text(size = axistext))
ggsave("runs/analysis/figures_R/meanY.pdf", plot = ts_ymean, width = 49, height = 12, limitsize=FALSE)

# --- 7. Turbidity vs Mean Y ---
cols = c("turb_ftu", "Y_mean")
finalsub_ab <- final_df %>% subset(Species %in% cols) %>% mutate(date = with_tz(date, tzone = "Asia/Tokyo"), day_label = format(date, "%m/%d"), hour_time = update(date, year = 2000, month = 1, mday = 1))
df_wide <- finalsub_ab %>% select(date, Species, Value) %>% subset(Value != 100) %>% pivot_wider(names_from = Species, values_from = Value)
y_vs_turb <- ggplot(df_wide, aes(x = turb_ftu, y = Y_mean)) +
  geom_point(aes(shape = Y_mean == 0), size = 3, alpha = 1) +  
  scale_shape_manual(values = c(16, 1), labels = c("Y > 0", "Y = 0"), name = "Condition") +
  theme_classic() +
  labs(x = "Turbidity", y = "Y_mean") +
  theme(legend.position = "none", axis.title = element_text(size = 18), axis.text  = element_text(size = 14)) +
  scale_x_continuous(breaks = seq(0, 10, by = 1)) +
  geom_vline(xintercept = c(0.5, 1, 2), linetype = "dashed", color = "grey50", linewidth = 0.8)
ggsave("runs/analysis/figures_R/YvsFTU.pdf", plot = y_vs_turb, width = 4, height = 4)

# --- 8. Family Level Ratio ---
# Replicating dfallsub_long_family logic
final_df_with_family <- final_df %>% left_join(dflabel %>% select(Species, family) %>% distinct(Species, .keep_all=TRUE), by = "Species")
dfallsub_long_family <- final_df_with_family %>% subset(!is.na(family)) %>% group_by(date, family, Species, Value) %>% summarise(Value = mean(Value, na.rm=TRUE), .groups="drop") %>% group_by(date, family) %>% summarise(Value = sum(Value, na.rm=TRUE), .groups="drop")

complete_df_family <- expand.grid(date = complete_dates, family = unique(dfallsub_long_family$family))
final_df_family <- complete_df_family %>% left_join(dfallsub_long_family, by = c("date", "family")) %>% mutate(Value = ifelse(is.na(Value), NA, Value))

cbp <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#8DD3C7", "#FB8072", "#80B1D3", "#FDB462", "#B3DE69", "#FCCDE5", "#BC80BD", "#CCEBC5", "#1B9E77", "#D95F02", "#7570B3", "#E7298A")
period1 <- c(as.POSIXct("2025-11-07 08:00:00", tz = "Asia/Tokyo"), as.POSIXct("2025-11-27 17:00:00", tz = "Asia/Tokyo"))
final_df_family2 <- final_df_family %>% subset(date >= period1[1] & date <= period1[2]) %>% mutate(day_label = format(date, "%m/%d", tz = "Asia/Tokyo"), hour_time = as.POSIXct(format(date, "%H:%M:%S"), format = "%H:%M:%S", tz = "Asia/Tokyo"))
df_ratio <- final_df_family2 %>% group_by(day_label, hour_time) %>% mutate(ratio = Value / sum(Value, na.rm = TRUE)) %>% ungroup()

c_ratio <- df_ratio %>% ggplot(aes(x = hour_time, y = ratio, fill = family)) +
  geom_col() +
  facet_wrap(~day_label, nrow = 1, strip.position = "bottom") +
  scale_x_datetime(date_labels = "%H:%M", breaks = "4 hours") +
  scale_fill_manual(values = cbp) +
  theme_classic() +
  theme(legend.position = "none", panel.spacing = unit(0.1, "lines"), strip.background = element_blank(), strip.placement = "outside", strip.text = element_text(size = 46, color = "black"), aspect.ratio = 2, axis.ticks = element_line(linewidth = 2), axis.ticks.length = unit(0.3, "cm"), axis.text.x = element_text(angle = 45, hjust = 1, size = 40), axis.title = element_text(size = axistitle), axis.text.y = element_text(size = 35)) 
ggsave("runs/analysis/figures_R/family_composition_ratio.pdf", plot = c_ratio, width = 49, height = 12, limitsize=FALSE)

print("Analysis and plotting completed successfully.")
