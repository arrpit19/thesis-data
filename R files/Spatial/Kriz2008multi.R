periodkriz2008 = c("2008-01-01", "2008-12-31")

clim_localkriz2008 <- getData("CLIMATE_LOCAL", site="bily_kriz", period = periodkriz2008)

site_datakriz<-getData("STAND", site="bily_kriz")

soil_datakriz<-getData("SOIL", site="bily_kriz", period=periodkriz)

tree_datakriz<-getData("TREE", site="bily_kriz")

data2004kriz<- tree_datakriz %>%
  filter(year == '2004')

unique_record_ids <- length(unique(data2004kriz$record_id))

fluxkriz2008 <-  getData("FLUX", site="bily_kriz", period = periodkriz2008)
GPPkriz2008 <- fluxkriz2008 %>% 
  group_by(year,mo) %>% 
  summarise(GPP = sum(gppDtVutRef_umolCO2m2s1)) %>% 
  select(GPP)



gppkriz2008<-GPPkriz2008$GPP*4.4*10^-4

#0.15*35cm 


library(dplyr)

# Calculate the mean monthly temperatures using dplyr
monthly_meanskriz2008 <- clim_localkriz2008 %>%
  group_by(mo) %>%
  summarise(MeanMaxTemp = mean(tmax_degC, na.rm = TRUE),
            MeanMinTemp = mean(tmin_degC, na.rm = TRUE),
            Monthlyrainfall=sum(p_mm,na.rm = TRUE),
            monthlymeansolar=mean(rad_Jcm2day, na.rm = TRUE),
            frostdays=sum(tmin_degC<0, na.rm = TRUE))

#to convert solar radiation unit from J/cm2 to MJ/m2.
monthly_meanskriz2008$monthlymeansolar<-monthly_meanskriz2008$monthlymeansolar*0.01

#removing the first column which is 'month number'
climatedatakriz2008<-monthly_meanskriz2008[,-1]

data2008kriz<- tree_datakriz %>%
  filter(year == '2008')

unique_record_ids2008 <- length(unique(data2008kriz$record_id))

climatedata_arraykriz2008 <- array(data = matrix(unlist(climatedatakriz2008), nrow = nrow(climatedatakriz2008), ncol = ncol(climatedatakriz2008)), dim = c(nrow(climatedatakriz2008), ncol(climatedatakriz), 1))

climatedata_arraykriz2008[,,1]

datasitekriz2008 <- data.frame(
  siteId = 2, 
  latitude = 49.3,    #latitude of the site
  nTrees = 375,      
  soilClass = 2,
  iAsw = 42,         #initial available water stored
  minAsw = 0,         #minimum available stored water  
  maxAsw =52.5 ,      #maximum available stored water = soil depth*((field capacity-wilting point)/100)
  poolFraction = 0,
  startAge = 11,      #?
  endAge = 12,
  startMonth = 1,
  wfI = 0.018,        # individual foliage biomass
  wrI = 0.019,       #root biomass
  wsI = 0.052,       #stem biomasss
  nThinnings = 0,
  climId = 2,
  fr = 0.5           #fertility rating
)

#this step is to combine the climate and site data dataaframes. if we want the put the input in neural networks as a combination of these two then this step should be used.

# Repeat the values from each column of df2 for each row in df1
repeated_valuesK2008 <- lapply(datasitekriz2008, function(x) rep(x, nrow(climatedatakriz2008)))

# Combine the dataframes horizontally
#joined_dfkriz2008 <- cbind(climatedatakriz2008, do.call(cbind, repeated_valuesK2008))

#newclimtestK2008<-joined_dfkriz2008[,-c(20:22)]

#kriztestmulti2008<-newclimtestK2008[,-c(6,11,13,16)]





#trying the model now
tryh2008<-r3pgn(siteData=datasitekriz2008, climate=climatedata_arraykriz2008, thinning = NULL, parameters = data_param[,2], outputs =26 )

# Repeat the values from each column of df2 for each row in df1
repeated_valueshy2008 <- lapply(datasite2004, function(x) rep(x, nrow(monthly_means2008)))

# Combine the dataframes horizontally
joined_dfhy2008 <- cbind(scale(monthly_means2008[,-1]), do.call(cbind, repeated_valueshy2008))

newclimtesthy2008<-joined_dfhy2008[,-c(20:22)]

hytestmulti2008<-newclimtesthy2008[,-c(6,11,13,16)]

hy<-as.data.frame(hytestmulti2008)

xtesthyy2008<-scale(hy[,c(1:5)])

#combined dataframe with climate and site dataframes for kriz
write.csv(hytestmulti2008[,-8],"xtesthyy200821.csv", row.names = FALSE)


