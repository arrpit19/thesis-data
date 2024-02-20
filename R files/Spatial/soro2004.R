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

clim_localsoro <- getData("CLIMATE_LOCAL", site="soro", period = periodkriz)

site_datasoro<-getData("STAND", site="soro")

soil_datasoro<-getData("SOIL", site="soro", period=periodkriz)

tree_datasoro<-getData("TREE", site="soro")

data2004soro<- tree_datasoro %>%
  filter(year == '2004')

unique_record_ids <- length(unique(data2004soro$record_id))


library(dplyr)

# Calculate the mean monthly temperatures using dplyr
monthly_meanssoro <- clim_localsoro %>%
  group_by(mo) %>%
  summarise(MeanMaxTemp = mean(tmax_degC, na.rm = TRUE),
            MeanMinTemp = mean(tmin_degC, na.rm = TRUE),
            Monthlyrainfall=sum(p_mm,na.rm = TRUE),
            monthlymeansolar=mean(rad_Jcm2day, na.rm = TRUE),
            frostdays=sum(tmin_degC<0, na.rm = TRUE))

#to convert solar radiation unit from J/cm2 to MJ/m2.
monthly_meanssoro$monthlymeansolar<-monthly_meanssoro$monthlymeansolar*0.01

#removing the first column which is 'month number' for r3pgn as it does not need month number as an input
climatedatasoro<-monthly_meanssoro[,-1]

#converting into a three dimensional array
climatedata_arraysoro <- array(data = matrix(unlist(climatedatasoro), nrow = nrow(climatedatasoro), ncol = ncol(climatedatasoro)), dim = c(nrow(climatedatasoro), ncol(climatedatasoro), 1))



#calculating biomass for the site data
biomasssoro<-site_datasoro%>%
  group_by(year)%>%
  summarise(foliagebiomass=sum(foliageBiomass_kgha, na.rm = TRUE),
            stembiomass=sum(stemBiomass_kgha, na.rm = TRUE),
            branbiomass=sum(branchesBiomass_kgha, na.rm=TRUE),
            rootbiomass=sum(rootBiomass_kgha, na.rm = TRUE))
dfkriz<-biomasskriz[8,]
#biomass unavailable

fluxsoro2004 <-  getData("FLUX", site="soro", period = periodkriz)
GPPsoro2004 <- fluxsoro2004 %>% 
  group_by(year,mo) %>% 
  summarise(GPP = sum(gppDtVutRef_umolCO2m2s1)) %>% 
  select(GPP)

gppsoro<-GPPsoro2004$GPP*4.4*10^-4


# Create a dataset with the specified column names and a numeric third dimension
datasitesoro <- data.frame(
  siteId = 4, 
  latitude = 55.49,    #latitude of the site
  nTrees = 307,      
  soilClass = 2,
  iAsw = 77,         #initial available water stored
  minAsw = 0,         #minimum available stored water  
  maxAsw =90.6 ,      #maximum available stored water = soil depth*((field capacity-wilting point)/100)
  poolFraction = 0,
  startAge = 60,      #?
  endAge = 61,
  startMonth = 1,
  wfI = 3.07,        # individual foliage biomass
  wrI = 41.1,       #stem plus branch biomass
  wsI = 3.684,       #root biomasss
  nThinnings = 0,
  climId = 4,
  fr = 0.5           #fertility rating
)

#biomass calculations using allometric equations.
#110 Italy a.D^b this is for foliage.   a= 0.00295   b= 2.43854

fls=(0.00295*29^2.43854)/1000   #kg-tonnes


#RT-  log(rt)= a+b·log(D) –1.66 2.54
logrt= -1.66+2.54*log10(29)


#ST= a·D^b·H^c    a= 0.00519 b=1.49634 c=2.10419    
st=(0.00519)*(45^1.496)*(25^2.104)
st=1.347 #tonnes

#combining climate and site data dataframes
# Repeat the values from each column of df2 for each row in df1 
repeated_valuessoro <- lapply(datasitesoro, function(x) rep(x, nrow(climatedatasoro)))

# Combine the dataframes horizontally
joined_dfsoro <- cbind(climatedatasoro, do.call(cbind, repeated_valuessoro))

newclimtestsoro2004<-joined_dfsoro[,-c(20:22)]

sorotestmulti2004<-newclimtestsoro2004[,-c(6,11,13,16)]








#trying the model now
trysoro<-r3pgn(siteData=datasitesoro, climate=climatedata_arraysoro, thinning = NULL, parameters = data_param[,2], outputs =26 )


#write.csv(climatedata, file = "climatetest.csv", row.names = FALSE)