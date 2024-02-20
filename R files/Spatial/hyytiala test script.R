library(threePGN)
library(tidyverse)
library(dplyr)
library(ProfoundData)
library(zoo)
library(lubridate)

# Download PROFOUND sqlite database to specified location and connect to database:
#downloadDatabase(location = "data")
#unzip("data/ProfoundData.zip")
vignette("ProfoundData")
setDB("ProfoundData.sqlite")
getDB()

# Explore the database
overview <- browseData()

# Hyytiala
sites <- overview$site[!unlist(apply(overview[,c("CLIMATE_LOCAL", "FLUX", "METEOROLOGICAL", "MODIS", "SOILTS")], 1,  function(x) any(0 %in% x)))]

#test period
periodtest = c("2008-01-01", "2008-12-31")

clim_local2008 <- getData("CLIMATE_LOCAL", site="hyytiala", period = periodtest)

site_data<-getData("STAND", site="hyytiala")

soil_data2008<-getData("SOIL", site="hyytiala")

tree_data<-getData("TREE", site="hyytiala")


library(dplyr)

# Calculate the mean monthly temperatures using dplyr
monthly_means2008 <- clim_local2008 %>%
  group_by(mo) %>%
  summarise(MeanMaxTemp = mean(tmax_degC, na.rm = TRUE),
            MeanMinTemp = mean(tmin_degC, na.rm = TRUE),
            Monthlyrainfall=sum(p_mm,na.rm = TRUE),
            monthlymeansolar=mean(rad_Jcm2day, na.rm = TRUE),
            frostdays=sum(tmin_degC<0, na.rm = TRUE))

#to convert solar radiation unit from J/cm2 to MJ/m2.
monthly_means2008$monthlymeansolar<-monthly_means2008$monthlymeansolar*0.01

#removing the first column which is 'month number'
climatedata2008<-monthly_means2008[,-1]

#converting into a three dimensional array
climatedata_array2008 <- array(data = matrix(unlist(climatedata2008), nrow = nrow(climatedata2008), ncol = ncol(climatedata2008)), dim = c(nrow(climatedata2008), ncol(climatedata2008), 1))

#data_climate[1,,]
#climatedata_array2008[1,,]

#calculating biomass for the site data
biomass<-site_data%>%
  group_by(year)%>%
  summarise(foliagebiomass=sum(foliageBiomass_kgha, na.rm = TRUE),
            stembiomass=sum(stemBiomass_kgha, na.rm = TRUE),
            branbiomass=sum(branchesBiomass_kgha, na.rm=TRUE),
            rootbiomass=sum(stumpCoarseRootBiomass_kgha, na.rm = TRUE))
df<-biomass[14,]     #for year 2008

library(dplyr)

unique_record_id_count <- tree_data %>%
  filter(year == 2008) %>%
  distinct(record_id) %>%
  nrow()




# Create a dataset with the specified column names and a numeric third dimension
datasite <- data.frame(
  siteId = 1, 
  latitude = 61.85,    #latitude of the site
  nTrees = 649,      
  soilClass = 2,
  iAsw = 85.1648,         #initial available water stored
  minAsw = 0,         #minimum available stored water  
  maxAsw =106.45 ,      #maximum available stored water = soil depth*((field capacity-wilting point)/100)
  poolFraction = 0,
  startAge = 47,      
  endAge = 48,
  startMonth = 1,
  wfI = 5.703,        # initial foliage biomass
  wrI = 22.47,       # root biomass
  wsI = 64.24,       #stem biomass
  nThinnings = 0,
  climId = 1,
  fr = 0.5           #fertility rating
)
# Repeat the values from each column of df2 for each row in df1
repeated_values <- lapply(datasite2008, function(x) rep(x, nrow(climatedata2008)))

# Combine the dataframes horizontally
joined_df <- cbind(climatedata2008, do.call(cbind, repeated_values))

newclimtest<-joined_df[,-c(20:22)]

hytestmulti2008<-newclimtest[,-c(6,11,13,16)]

#trying the model now
firsttry2008<-r3pgn(siteData=datasite2008, climate=climatedata_array2008, thinning = NULL, parameters = data_param[,2], outputs =26 )

kl<-firsttry2008[["output"]]

write.csv(kl, "pred2008uncali.csv", row.names = FALSE)
jj<-read.csv("pred2008uncali.csv")


#flux to get gpp
fluxtest <-  getData("FLUX", site="hyytiala", period = periodtest)

GPPtestsum <- fluxtest %>% 
  group_by(year,mo) %>% 
  summarise(GPP = sum(gppDtVutRef_umolCO2m2s1)) %>% 
  select(GPP)

gpptestnew<-GPPtestsum$GPP*4.4*10^-4     #to convert to tonnes/hectares.

maeuncali <- sum(abs(gpptestnew - jj))/length(jj)   #uncalibrated model mean abso error.


write.csv(climatedata2008, file = "mutliclimatetestHY.csv", row.names = FALSE)

write.csv(firsttry2008[["output"]], file="gppmodelpredictionn2008.csv", row.names=FALSE)

