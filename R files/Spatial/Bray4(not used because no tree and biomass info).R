#multiple sites

library(threePGN)
library(tidyverse)
library(dplyr)
library(ProfoundData)
library(zoo)
library(lubridate)

# Download PROFOUND sqlite database to specified location and connect to database:
#   downloadDatabase(location = "data")
#   unzip("data/ProfoundData.zip")
vignette("ProfoundData")
setDB("ProfoundData.sqlite")
getDB()

# Explore the database
overview <- browseData()

sites <- overview$site[!unlist(apply(overview[,c("CLIMATE_LOCAL", "FLUX", "METEOROLOGICAL", "MODIS", "SOILTS")], 1,  function(x) any(0 %in% x)))]


# Define a period (see Elias Schneider)
periodkriz = c("2004-01-01", "2004-12-31")

clim_localbray <- getData("CLIMATE_LOCAL", site="le_bray", period = periodkriz)

site_databray<-getData("STAND", site="le_bray")

soil_databray<-getData("SOIL", site="le_bray", period=periodkriz)

#tree_databray<-getData("TREE", site="le_bray")
#treedata also not available
data2004cole<- tree_datakriz %>%
  filter(year == '2004')

unique_record_ids <- length(unique(data2004kriz$record_id))


library(dplyr)

# Calculate the mean monthly temperatures using dplyr
monthly_meansbray <- clim_localbray %>%
  group_by(mo) %>%
  summarise(MeanMaxTemp = mean(tmax_degC, na.rm = TRUE),
            MeanMinTemp = mean(tmin_degC, na.rm = TRUE),
            Monthlyrainfall=sum(p_mm,na.rm = TRUE),
            monthlymeansolar=mean(rad_Jcm2day, na.rm = TRUE),
            frostdays=sum(tmin_degC<0, na.rm = TRUE))

#to convert solar radiation unit from J/cm2 to MJ/m2.
monthly_meansbray$monthlymeansolar<-monthly_meansbray$monthlymeansolar*0.01

#removing the first column which is 'month number'
climatedatabray<-monthly_meansbray[,-1]

#converting into a three dimensional array
climatedata_arraybray <- array(data = matrix(unlist(climatedatabray), nrow = nrow(climatedatabray), ncol = ncol(climatedatabray)), dim = c(nrow(climatedatabray), ncol(climatedatabray), 1))


#calculating biomass for the site data
biomassbray<-site_databray%>%
  group_by(year)%>%
  summarise(foliagebiomass=sum(foliageBiomass_kgha, na.rm = TRUE),
            stembiomass=sum(stemBiomass_kgha, na.rm = TRUE),
            branbiomass=sum(branchesBiomass_kgha, na.rm=TRUE),
            rootbiomass=sum(rootBiomass_kgha, na.rm = TRUE))
dfkriz<-biomasskriz[8,]
#unavailable




# Create a dataset with the specified column names and a numeric third dimension
datasitebray <- data.frame(
  siteId = 4, 
  latitude = 44.72,    #latitude of the site
  nTrees = ,      
  soilClass = 1,
  iAsw = 80,         #initial available water stored
  minAsw = 0,         #minimum available stored water  
  maxAsw =100 ,      #maximum available stored water = soil depth*((field capacity-wilting point)/100)
  poolFraction = 0,
  startAge = 34,      #?
  endAge = 35,
  startMonth = 1,
  wfI = 10.015,        # individual foliage biomass
  wrI = 20.014,       #stem plus branch biomass
  wsI = 30.052,       #root biomasss
  nThinnings = 0,
  climId = 1,
  fr = 0.5           #fertility rating
)

#trying the model now
trybray<-r3pgn(siteData=datasitebray, climate=climatedata_arraybray, thinning = NULL, parameters = data_param[,2], outputs =26 )


#write.csv(climatedata, file = "climatetest.csv", row.names = FALSE)


