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


# Define a period
periodkriz = c("2004-01-01", "2004-12-31")

clim_localcole <- getData("CLIMATE_LOCAL", site="collelongo", period = periodkriz)

site_datacole<-getData("STAND", site="collelongo")

soil_datacole<-getData("SOIL", site="collelongo", period=periodkriz)

tree_datacole<-getData("TREE", site="collelongo")

data2004cole<- tree_datakriz %>%
  filter(year == '2004')

unique_record_ids <- length(unique(data2004kriz$record_id))


fluxcole2004 <-  getData("FLUX", site="collelongo", period = periodkriz)
GPPcole2004 <- fluxcole2004 %>% 
  group_by(year,mo) %>% 
  summarise(GPP = sum(gppDtVutRef_umolCO2m2s1)) %>% 
  select(GPP)


#convert to tonnes/hectares
gppcole2004<-GPPcole2004$GPP*4.4*10^-4



library(dplyr)

# Calculate the mean monthly temperatures using dplyr
monthly_meanscole <- clim_localcole %>%
  group_by(mo) %>%
  summarise(MeanMaxTemp = mean(tmax_degC, na.rm = TRUE),
            MeanMinTemp = mean(tmin_degC, na.rm = TRUE),
            Monthlyrainfall=sum(p_mm,na.rm = TRUE),
            monthlymeansolar=mean(rad_Jcm2day, na.rm = TRUE),
            frostdays=sum(tmin_degC<0, na.rm = TRUE))

#to convert solar radiation unit from J/cm2 to MJ/m2.
monthly_meanscole$monthlymeansolar<-monthly_meanscole$monthlymeansolar*0.01

#removing the first column which is 'month number'
climatedatacole<-monthly_meanscole[,-1]

#converting into a three dimensional array
climatedata_arraycole <- array(data = matrix(unlist(climatedatacole), nrow = nrow(climatedatacole), ncol = ncol(climatedatacole)), dim = c(nrow(climatedatacole), ncol(climatedatacole), 1))

data_climate[1,,]
climatedata_array1[1,,]

#calculating biomass for the site data
biomasscole<-site_datacole%>%
  group_by(year)%>%
  summarise(foliagebiomass=sum(foliageBiomass_kgha, na.rm = TRUE),
            stembiomass=sum(stemBiomass_kgha, na.rm = TRUE),
            branbiomass=sum(branchesBiomass_kgha, na.rm=TRUE),
            rootbiomass=sum(rootBiomass_kgha, na.rm = TRUE))
dfkriz<-biomasskriz[8,]


#dbh-45cm
#height- 25m

#FL    foliage biomass
#RT    root biomass      
#ST    Steam biomass

#Allometric equations to estimate biomass using DBH and height of a tree from the stand 


#110 Italy- a.D^b this is for foliage.   a= 0.00295   b= 2.43854

 fl=(0.00295*45^2.43854)/1000   #kg-tonnes


#RT-  log(rt)= a+b·log(D) –1.66 2.54
 logrt= -1.66+2.54*log10(45)
  

#ST= a·D^b·H^c    a= 0.00519 b=1.49634 c=2.10419    
st=(0.00519)*(45^1.496)*(25^2.104)
st=1.347 #tonnes

#800*(41.4-24.8)/100
# biomass are multiplied by number of trees to get the entire individual biomass of the stand/site


# Create a dataset with the specified column names and a numeric third dimension
datasitecole <- data.frame(
  siteId = 3, 
  latitude = 41.85,    #latitude of the site
  nTrees = 165,      
  soilClass = 2,
  iAsw = 112.4,         #initial available water stored
  minAsw = 0,         #minimum available stored water  
  maxAsw =132.6 ,      #maximum available stored water = soil depth*((field capacity-wilting point)/100)
  poolFraction = 0,
  startAge = 14,    
  endAge = 15,
  startMonth = 1,
  wfI = 4.95,        # individual foliage biomass
  wrI = 56.9,       #root  biomass
  wsI = 214.5,       #stem biomasss
  nThinnings = 0,
  climId = 3,
  fr = 0.5           #fertility rating
)

#to combine climate and site dataframes
# Repeat the values from each column of df2 for each row in df1
repeated_valuescole <- lapply(datasitecole, function(x) rep(x, nrow(climatedatacole)))

# Combine the dataframes horizontally
joined_dfcole <- cbind(climatedatacole, do.call(cbind, repeated_valuescole))

newclimtestcole<-joined_dfcole[,-c(20:22)]

coletestmulti2004<-newclimtestcole[,-c(6,11,13,16)]




#trying the model now
trycole<-r3pgn(siteData=datasitecole, climate=climatedata_arraycole, thinning = NULL, parameters = data_param[,2], outputs =26 )



#other steps were carried out during the same time in other files 


