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

# treaining period
periodtrain = c("2001-01-01", "2006-12-31")

clim_localtrain <- getData("CLIMATE_LOCAL", site="hyytiala", period = periodtrain)

clim_localtrain$new_month <- (clim_localtrain$year - min(clim_localtrain$year)) * 12 + clim_localtrain$mo


# Calculate the mean monthly temperatures using dplyr
monthly_meanstrain <- clim_localtrain %>%
  group_by(new_month) %>%
  summarise(MeanMaxTemp = mean(tmax_degC, na.rm = TRUE),
            MeanMinTemp = mean(tmin_degC, na.rm = TRUE),
            Monthlyrainfall=sum(p_mm,na.rm = TRUE),
            monthlymeansolar=mean(rad_Jcm2day, na.rm = TRUE),
            frostdays=sum(tmin_degC<0, na.rm = TRUE))

#to convert from J/cm2 to MJ/m2
monthly_meanstrain$monthlymeansolar<-monthly_meanstrain$monthlymeansolar*0.01

monthly_meanstrainclimate<-monthly_meanstrain

#removing the first column which is 'month number' because climate input in r3pgn does not neeed month number as a column
monthly_meanstrainclimate<-monthly_meanstrainclimate[,-1]

#adding months and years colomns to monthlymeans_train (to be used later during calibration of the r3pgn model)

newtraindata<-monthly_meanstrainclimate

newtraindata$months <- rep(1:12, times = 6)  # Repeats months 1-12 for 7 years
newtraindata$years <- rep(2001:2006, each = 12)  # Repeats years 2001-2007 for 12 months each

#creating a climate array for r3pgn
climatedata_arraytrain <- array(data = matrix(unlist(monthly_meanstrainclimate), nrow = nrow(monthly_meanstrainclimate), ncol = ncol(monthly_meanstrainclimate)), dim = c(nrow(monthly_meanstrainclimate), ncol(monthly_meanstrainclimate), 1))

#calculating biomass for the site data
biomass<-site_data%>%
  group_by(year)%>%
  summarise(foliagebiomass=sum(foliageBiomass_kgha, na.rm = TRUE),
            stembiomass=sum(stemBiomass_kgha, na.rm = TRUE),
            branbiomass=sum(branchesBiomass_kgha, na.rm=TRUE),
            rootbiomass=sum(stumpCoarseRootBiomass_kgha, na.rm = TRUE))

# Assuming your dataframe is named "df"
df_filtered <- biomass[biomass$year == 2001,]
df_filtered2004 <- biomass[biomass$year == 2004,]

biomass_sums <- colSums(df_filtered[, c("foliagebiomass", "stembiomass", "branbiomass", "rootbiomass")])

print(biomass_sums)


trainr3pgn<-r3pgn(siteData=datasite, climate=climatedata_arraytrain, thinning = NULL, parameters = data_param[,2], outputs =26 )


fluxtrain <-  getData("FLUX", site="hyytiala", period = periodtrain)
GPPtrain <- fluxtrain %>% 
  group_by(year,mo) %>% 
  summarise(GPP = mean(gppDtVutRef_umolCO2m2s1)) %>% 
  select(GPP)

GPPtrainsum <- fluxtrain %>% 
  group_by(year,mo) %>% 
  summarise(GPP = sum(gppDtVutRef_umolCO2m2s1)) %>% 
  select(GPP)

gppnew<-GPPtrainsum$GPP*4.4*10^-4          # conversion to tonnes/hectares

gpppretrain<-gppnew[c(1:12)]    #gpp for pre train
gppNAS<-gppnew[c(13:36)]        #gpp for NAS 
gpptrain<-gppnew[c(37:72)]       #gpp for train

write.csv(gpppretrain, file = "ypretrain.csv", row.names = FALSE)
write.csv(gppNAS,file="YNAS.csv", row.names = FALSE)
write.csv(gpptrain,file="Ytrain.csv", row.names = FALSE)




gppactual<-GPPtrain$GPP*4.4*10^-4

#gppaug<-gppnew[-c(73:84)]

#auggpp<-data.frame(gpp=gppaug,
                   #months=newtraindata$months,
                   #year=newtraindata$years)


write.csv(gppnew, file = "gpptrainmonthlysum.csv", row.names = FALSE)

climate_scaled<-scale(monthly_meanstrainclimate)

write.csv(monthly_meanstrainclimate, file = "climatetrain.csv", row.names = FALSE)
write.csv(climate_scaled, file = "climatetrain1.csv", row.names = FALSE)


GPPtrain1<-trainr3pgn[["output"]]

gpptrain11<-GPPtrain1[,,1]


#climate data for NAS 
nasdata<-monthly_meanstrainclimate[c(13:36),]

#scaling that data
nasdatascaled<-scale(nasdata)

write.csv(nasdata, file="nasdataSS.csv", row.names = FALSE )
write.csv(nasdatascaled, file="nasdataSSscaled.csv", row.names = FALSE)

#climate data for train
traindataSS<-monthly_meanstrainclimate[c(37:72),]

#scaling the training data
traindatascaledss<-scale(traindataSS)

write.csv(traindataSS, file="traindataSS.csv", row.names=FALSE)
write.csv(traindatascaledss, file="traindataSSscaled.csv", row.names=FALSE)

pretraindataSS<-monthly_meanstrainclimate[c(1:12),]
pretraindatascaledss<-scale(pretraindataSS)

write.csv(pretraindataSS, file="pretraindataSS.csv", row.names=FALSE)
write.csv(pretraindatascaledss, file="pretraindataSSscaled.csv", row.names=FALSE)




#for hyytiala data Spatial experiment for NAS (year=2004)

# Create a dataset with the specified column names
datasite2004 <- data.frame(
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
  wfI = 5.903,        # individual foliage biomass
  wrI = 21.7,       # root biomass
  wsI = 68.24,       #stem biomass
  nThinnings = 0,
  climId = 1,
  fr = 0.5           #fertility rating
)


#making the data where climate is combine with the site data to make one dataframe. this was tried and it produced huge error so , it was not used.
# Repeat the values from each column of df2 for each row in df1
repeated_values2004 <- lapply(datasite2004, function(x) rep(x, nrow(monthly_meanstrainclimate[c(37:48),])))

# Combine the dataframes horizontally
joined_df2004 <- cbind(monthly_meanstrainclimate[c(37:48),], do.call(cbind, repeated_values2004))

newclimtest2004<-joined_df2004[,-c(20:22)]

hytestmulti2004<-newclimtest2004[,-c(6,11,13,16)]

