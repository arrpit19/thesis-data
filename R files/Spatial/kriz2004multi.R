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
# Use only stands with all required data sets available and pick a site. 
# Hyytiala
sites <- overview$site[!unlist(apply(overview[,c("CLIMATE_LOCAL", "FLUX", "METEOROLOGICAL", "MODIS", "SOILTS")], 1,  function(x) any(0 %in% x)))]


# Define a period (see Elias Schneider)
periodkriz = c("2004-01-01", "2004-12-31")

clim_localkriz <- getData("CLIMATE_LOCAL", site="bily_kriz", period = periodkriz)

site_datakriz<-getData("STAND", site="bily_kriz")

soil_datakriz<-getData("SOIL", site="bily_kriz", period=periodkriz)

tree_datakriz<-getData("TREE", site="bily_kriz")

data2004kriz<- tree_datakriz %>%
  filter(year == '2004')

unique_record_ids <- length(unique(data2004kriz$record_id))

fluxkriz2004 <-  getData("FLUX", site="bily_kriz", period = periodkriz)
GPPkriz2004 <- fluxkriz2004 %>% 
  group_by(year,mo) %>% 
  summarise(GPP = sum(gppDtVutRef_umolCO2m2s1)) %>% 
  select(GPP)



gppkriz<-GPPkriz2004$GPP*4.4*10^-4

#0.15*35cm 


library(dplyr)

# Calculate the mean monthly temperatures using dplyr
monthly_meanskriz <- clim_localkriz %>%
  group_by(mo) %>%
  summarise(MeanMaxTemp = mean(tmax_degC, na.rm = TRUE),
            MeanMinTemp = mean(tmin_degC, na.rm = TRUE),
            Monthlyrainfall=sum(p_mm,na.rm = TRUE),
            monthlymeansolar=mean(rad_Jcm2day, na.rm = TRUE),
            frostdays=sum(tmin_degC<0, na.rm = TRUE))

#to convert solar radiation unit from J/cm2 to MJ/m2.
monthly_meanskriz$monthlymeansolar<-monthly_meanskriz$monthlymeansolar*0.01

#removing the first column which is 'month number'
climatedatakriz<-monthly_meanskriz[,-1]

#converting into a three dimensional array
climatedata_arraykriz <- array(data = matrix(unlist(climatedatakriz), nrow = nrow(climatedatakriz), ncol = ncol(climatedatakriz)), dim = c(nrow(climatedatakriz), ncol(climatedatakriz), 1))

#data_climate[1,,]
#climatedata_array1[1,,]

#calculating biomass for the site data
biomasskriz<-site_datakriz%>%
  group_by(year)%>%
  summarise(foliagebiomass=sum(foliageBiomass_kgha, na.rm = TRUE),
            stembiomass=sum(stemBiomass_kgha, na.rm = TRUE),
            branbiomass=sum(branchesBiomass_kgha, na.rm=TRUE),
            rootbiomass=sum(rootBiomass_kgha, na.rm = TRUE))
dfkriz<-biomasskriz[8,]





# Create a dataset with the specified column names and a numeric third dimension
datasitekriz <- data.frame(
  siteId = 2, 
  latitude = 49.3,    #latitude of the site
  nTrees = 414,      
  soilClass = 2,
  iAsw = 42,         #initial available water stored
  minAsw = 0,         #minimum available stored water  
  maxAsw =52.5 ,      #maximum available stored water = soil depth*((field capacity-wilting point)/100)
  poolFraction = 0,
  startAge = 7,      #?
  endAge = 8,
  startMonth = 1,
  wfI = 0.015,        # individual foliage biomass
  wrI = 0.014,       #root biomass
  wsI = 0.036,       #stem biomasss
  nThinnings = 0,
  climId = 2,
  fr = 0.5           #fertility rating
)


#this step is to combine the climate and site data dataaframes. if we want the put the input in neural networks as a combination of these two then this step should be used.
# Repeat the values from each column of df2 for each row in df1
repeated_valueskriz <- lapply(datasitekriz, function(x) rep(x, nrow(climatedatakriz)))

# Combine the dataframes horizontally
joined_dfkriz <- cbind(climatedatakriz, do.call(cbind, repeated_valueskriz))

#newclimtestkriz<-joined_dfkriz[,-c(20:22)]

#kriztestmulti2004<-newclimtestkriz[,-c(6,11,13,16)]





#trying the model now
trykriz<-r3pgn(siteData=datasitekriz, climate=climatedata_arraykriz, thinning = NULL, parameters = data_param[,2], outputs =26 )


write.csv(climatedata, file = "climatekriz2004.csv", row.names = FALSE)


