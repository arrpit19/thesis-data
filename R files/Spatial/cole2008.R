
# Define a period 
periodcole2008 = c("2008-01-01", "2008-12-31")

clim_localcole2008 <- getData("CLIMATE_LOCAL", site="collelongo", period = periodcole2008)

site_datacole<-getData("STAND", site="collelongo")

soil_datacole<-getData("SOIL", site="collelongo", period=periodcole2008)

tree_datacole<-getData("TREE", site="collelongo")

data2008cole<- tree_datacole %>%
  filter(year == '2008')

unique_record_ids <- length(unique(data2008cole$record_id))


fluxcole2008 <-  getData("FLUX", site="collelongo", period = periodcole2008)
GPPcole2008 <- fluxcole2008 %>% 
  group_by(year,mo) %>% 
  summarise(GPP = sum(gppDtVutRef_umolCO2m2s1)) %>% 
  select(GPP)



gppcole2008<-GPPcole2008$GPP*4.4*10^-4

#mean dbh to be used for allometric equations 
dbh<-mean(tree_datacole$height1_m[c(691:528)])

library(dplyr)

# Calculate the mean monthly temperatures using dplyr
monthly_meanscole2008 <- clim_localcole2008 %>%
  group_by(mo) %>%
  summarise(MeanMaxTemp = mean(tmax_degC, na.rm = TRUE),
            MeanMinTemp = mean(tmin_degC, na.rm = TRUE),
            Monthlyrainfall=sum(p_mm,na.rm = TRUE),
            monthlymeansolar=mean(rad_Jcm2day, na.rm = TRUE),
            frostdays=sum(tmin_degC<0, na.rm = TRUE))

#to convert solar radiation unit from J/cm2 to MJ/m2.
monthly_meanscole2008$monthlymeansolar<-monthly_meanscole2008$monthlymeansolar*0.01

#removing the first column which is 'month number' because climate input in r3pgn does not neeed month number as a column
climatedatacole2008<-monthly_meanscole2008[,-1]

#converting into a three dimensional array
climatedata_arraycole2008 <- array(data = matrix(unlist(climatedatacole2008), nrow = nrow(climatedatacole2008), ncol = ncol(climatedatacole2008)), dim = c(nrow(climatedatacole2008), ncol(climatedatacole2008), 1))


#calculating biomass for the site data
biomasscole<-site_datacole%>%
  group_by(year)%>%
  summarise(foliagebiomass=sum(foliageBiomass_kgha, na.rm = TRUE),
            stembiomass=sum(stemBiomass_kgha, na.rm = TRUE),
            branbiomass=sum(branchesBiomass_kgha, na.rm=TRUE),
            rootbiomass=sum(rootBiomass_kgha, na.rm = TRUE))
dfkriz<-biomasskriz[8,]


#dbh-22.6cm
#height- 18.8m

#FL    foliage  biomass
#RT    root     biomass  
#ST   stem       biomass  


#110 Italy a.D^b this is for foliage.   a= 0.00295   b= 2.43854

flc=(0.00295*22.6^2.43854)/1000   #kg-tonnes


#RT-  log(rt)= a+b·log(D) –1.66 2.54
logrt= -1.66+2.54*log10(22.6)


#ST= a·D^b·H^c    a= 0.00519 b=1.49634 c=2.10419    
st=(0.00519)*(22.6^1.496)*(18^2.104)
st=1.347 #tonnes

#800*(41.4-24.8)/100

# biomass are multiplied by number of trees to get the entire individual biomass of the stand/site
#0.05*162
#0.06*162
#0.24*162

# Create a dataset with the specified column names and a numeric third dimension
datasitecole2008 <- data.frame(
  siteId = 3, 
  latitude = 41.85,    #latitude of the site
  nTrees = 162,      
  soilClass = 2,
  iAsw = 112.4,         #initial available water stored
  minAsw = 0,         #minimum available stored water  
  maxAsw =132.6 ,      #maximum available stored water = soil depth*((field capacity-wilting point)/100)
  poolFraction = 0,
  startAge = 18,      #?
  endAge = 19,
  startMonth = 1,
  wfI = 8.1,        # individual foliage biomass
  wrI = 9.72,       #stem  biomass
  wsI = 38.88,       #root biomasss
  nThinnings = 0,
  climId = 3,
  fr = 0.5           #fertility rating
)

#to combine climate and site dataframes
# Repeat the values from each column of df2 for each row in df1
repeated_valuescole2008 <- lapply(datasitecole2008, function(x) rep(x, nrow(climatedatacole2008)))

# Combine the dataframes horizontally
joined_dfcole2008 <- cbind(climatedatacole2008, do.call(cbind, repeated_valuescole2008))

newclimtestcole2008<-joined_dfcole2008[,-c(20:22)]

coletestmulti2008<-newclimtestcole2008[,-c(6,11,13,16)]






#trying the model now
trycole<-r3pgn(siteData=datasitecole2008, climate=climatedata_arraycole2008, thinning = NULL, parameters = data_param[,2], outputs =26 )



#write.csv(climatedata, file = "climatetest.csv", row.names = FALSE)
