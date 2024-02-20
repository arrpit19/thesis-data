periodsoro2008 = c("2008-01-01", "2008-12-31")

clim_localsoro2008 <- getData("CLIMATE_LOCAL", site="soro", period = periodsoro2008)

site_datasoro<-getData("STAND", site="soro")

soil_datasoro<-getData("SOIL", site="soro", period=periodkriz)

tree_datasoro<-getData("TREE", site="soro")

data2008soro<- tree_datasoro %>%
  filter(year == '2008')

unique_record_ids <- length(unique(data2004soro$record_id))


library(dplyr)

# Calculate the mean monthly temperatures using dplyr
monthly_meanssoro2008 <- clim_localsoro2008 %>%
  group_by(mo) %>%
  summarise(MeanMaxTemp = mean(tmax_degC, na.rm = TRUE),
            MeanMinTemp = mean(tmin_degC, na.rm = TRUE),
            Monthlyrainfall=sum(p_mm,na.rm = TRUE),
            monthlymeansolar=mean(rad_Jcm2day, na.rm = TRUE),
            frostdays=sum(tmin_degC<0, na.rm = TRUE))

#to convert solar radiation unit from J/cm2 to MJ/m2.
monthly_meanssoro2008$monthlymeansolar<-monthly_meanssoro2008$monthlymeansolar*0.01

#removing the first column which is 'month number'
climatedatasoro2008<-monthly_meanssoro2008[,-1]

#converting into a three dimensional array
climatedata_arraysoro2008 <- array(data = matrix(unlist(climatedatasoro2008), nrow = nrow(climatedatasoro2008), ncol = ncol(climatedatasoro2008)), dim = c(nrow(climatedatasoro2008), ncol(climatedatasoro2008), 1))


#calculating biomass for the site data
biomasssoro<-site_datasoro%>%
  group_by(year)%>%
  summarise(foliagebiomass=sum(foliageBiomass_kgha, na.rm = TRUE),
            stembiomass=sum(stemBiomass_kgha, na.rm = TRUE),
            branbiomass=sum(branchesBiomass_kgha, na.rm=TRUE),
            rootbiomass=sum(rootBiomass_kgha, na.rm = TRUE))
dfkriz<-biomasskriz[8,]
#biomass unavailable

fluxsoro2008 <-  getData("FLUX", site="soro", period = periodsoro2008)
GPPsoro2008 <- fluxsoro2008 %>% 
  group_by(year,mo) %>% 
  summarise(GPP = sum(gppDtVutRef_umolCO2m2s1)) %>% 
  select(GPP)

gppsoro2008<-GPPsoro2008$GPP*4.4*10^-4


# Create a dataset with the specified column names and a numeric third dimension
datasitesoro2008 <- data.frame(
  siteId = 4, 
  latitude = 55.49,    #latitude of the site
  nTrees = 288,      
  soilClass = 2,
  iAsw = 77,         #initial available water stored
  minAsw = 0,         #minimum available stored water  
  maxAsw =90.6 ,      #maximum available stored water = soil depth*((field capacity-wilting point)/100)
  poolFraction = 0,
  startAge = 64,      #?
  endAge = 68,
  startMonth = 1,
  wfI = 2.88,        # individual foliage biomass
  wrI = 3.456,       #stem plus branch biomass
  wsI = 162.432,       #root biomasss
  nThinnings = 0,
  climId = 4,
  fr = 0.5           #fertility rating
)

dbh<-mean(data2008soro$height1_m)
#dbh=28.34
#height=22.97
#110 Italy a.D^b this is for foliage.   a= 0.00295   b= 2.43854

fls=(0.00295*28.34^2.43854)/1000   #kg-tonnes
flsoro=0.01*288

#RT-  log(rt)= a+b·log(D) –1.66 2.54
logrt= -1.66+2.54*log10(29)
rt=e^2.0539 
rt=345.9 #kg
rt=0.345 #tonnes 



#ST= a·D^b·H^c    a= 0.00519 b=1.49634 c=2.10419    
st=(0.00519)*(28.34^1.496)*(22.97^2.104)
st=1.347 #tonnes

# Repeat the values from each column of df2 for each row in df1
repeated_valuessoro2008 <- lapply(datasitesoro2008, function(x) rep(x, nrow(climatedatasoro2008)))

# Combine the dataframes horizontally
joined_dfsoro2008 <- cbind(climatedatasoro2008, do.call(cbind, repeated_valuessoro2008))

newclimtestsoro2008<-joined_dfsoro2008[,-c(20:22)]

sorotestmulti2008<-newclimtestsoro2008[,-c(6,11,13,16)]




#trying the model now
trysoro2008<-r3pgn(siteData=datasitesoro2008, climate=climatedata_arraysoro2008, thinning = NULL, parameters = data_param[,2], outputs =26 )
