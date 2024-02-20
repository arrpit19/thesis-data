require(mgcv)
require(lubridate)
require(mvtnorm)
require(ggplot2)

#newtraindata and gppnew are used from hyytiala 2001:2006 script

newtraindata1<-newtraindata
newtraindata1$gpp<-gppnew

fmT = gam(MeanMaxTemp ~ s(months, by=years, bs = "cc"), data=newtraindata1)
fmTmin = gam(MeanMinTemp ~ s(months, by=years, bs = "cc"), data=newtraindata1)
fmPrec = gam(Monthlyrainfall ~ s(months, by=years, bs = "cc"), data=newtraindata1)
fmsolrad = gam(monthlymeansolar ~ s(months, by=years, bs = "cc"), data=newtraindata1)
fmfrost = gam(frostdays ~ s(months, by=years, bs = "cc"), data=newtraindata1)
fmgpp=gam(gpp~s(months, by=years, bs="cc"), data=newtraindata1)



summary(fmT)
summary(fmTmin)
summary(fmPrec)
summary(fmsolrad)
summary(fmfrost)

plot(fmT$residuals)
plot(fmTmin$residuals)
plot(fmPrec$residuals)
plot(fmsolrad$residuals)
plot(fmfrost$residuals)
plot(fmgpp$residuals)

# predicting till 1440 months
T_hat = predict(fmT, data.frame(months = 1:1440, years=2001:2006))
Tmin_hat = predict(fmTmin, data.frame(months = 1:1440, years=2001:2006))
Prec_hat = predict(fmPrec, data.frame(months = 1:1440, years=2001:2006))
Solar_hat = predict(fmsolrad, data.frame(months = 1:1440, years=2001:2006))
Frost_hat = predict(fmfrost, data.frame(months = 1:1440, years=2001:2006))
gpp_hat=predict(fmgpp, data.frame(months = 1:1440, years=2001:2006))


plot(1:1440, T_hat, type="l")
plot(1:1440, Tmin_hat, type="l")
plot(1:1440, Prec_hat, type="l")
plot(1:1440, Solar_hat, type="l")
plot(1:1440, Frost_hat, type="l")
plot(1:1440, gpp_hat, tyle="l")


#creating a residual matrix 
res_mat = data.frame(Tmax=fmT$residuals, 
                     Tmin=fmTmin$residuals, 
                     Prec=fmPrec$residuals, 
                     Solrad=fmsolrad$residuals, 
                     Frost = fmfrost$residuals,
                     gpp=fmgpp$residuals)

plot(res_mat)
summary(res_mat)

#covariance matrix
cov_mat = cov(res_mat)

#creating noise
noise = rmvnorm(1440, mean=rep(0, length=ncol(res_mat)), sigma=cov_mat)

summary(noise)
plot(1:1440, noise[,3]) 


T_hat = T_hat + noise[,1]
Tmin_hat = Tmin_hat + noise[,2]
Prec_hat = Prec_hat + noise[,3]
Solar_hat = Solar_hat + noise[,4]
Frost_hat = Frost_hat + noise[,5]
gpp_hat=gpp_hat+noise[,6]


#scaling the frost_hat data to 0-31 range because frost days can only be between 0-31
new_min<-0
new_max<-31
scaled_data <- ((Frost_hat - min(Frost_hat)) / (max(Frost_hat) - min(Frost_hat))) * (new_max - new_min) + new_min

#putting values to 0 which are negative 
gpp_hat[which(gpp_hat<0)] = 0
Prec_hat[which(Prec_hat<0)] = 0
Solar_hat[which(Solar_hat<0)] = 0


#creating the aug data
augmenteddata<-data.frame(MeanMaxTemp=T_hat,
                          MeanMinTemp=Tmin_hat,
                          monthlyrainfall=Prec_hat,
                          monthlymeansolar=Solar_hat,
                          frostdays=scaled_data)



#scaling the data
augclim_scaled<-scale(augmenteddata)

#gpp
gppaug<-as.data.frame(gpp_hat)


#120 for pre training 
#360 for NAS
#960 for training

augpretrain<-augclim_scaled[c(1:120),]
augNAStemp<-augclim_scaled[c(121:480),]
augtraintemp<-augclim_scaled[c(481:1440),]



write.csv(augpretrain, file = "augpretrain.csv", row.names = FALSE)
write.csv(gppaug[c(1:120),], file = "auggpppretrain.csv", row.names = FALSE)


write.csv(augNAStemp, file = "augclim.csv", row.names = FALSE)
write.csv(gppaug[c(121:480),], file = "auggppNAS.csv", row.names = FALSE)

write.csv(augtraintemp, file = "augtraintemp.csv", row.names = FALSE)
write.csv(gppaug[c(481:1440),], file = "auggpptrain.csv", row.names = FALSE)


augmenteddataarray <- array(data = matrix(unlist(augmenteddata), nrow = nrow(augmenteddata), ncol = ncol(augmenteddata)), dim = c(nrow(augmenteddata), ncol(augmenteddata), 1))

datasiteaug <- data.frame(
  siteId = 1, 
  latitude = 61.85,    #latitude of the site
  nTrees = 649,      
  soilClass = 2,
  iAsw = 85.1648,         #initial available water stored
  minAsw = 0,         #minimum available stored water  
  maxAsw =106.45 ,      #maximum available stored water = soil depth*((field capacity-wilting point)/100)
  poolFraction = 0,
  startAge = 40,      
  endAge = 46,
  startMonth = 1,
  wfI = 5.703,        # individual foliage biomass
  wrI = 22.47,       # root biomass
  wsI = 64.24,       #stem biomass
  nThinnings = 0,
  climId = 1,
  fr = 0.5           #fertility rating
)

augr3pgngpp<-r3pgn(siteData=datasiteaug, climate=augmenteddataarray, thinning = NULL, parameters = data_param[,2], outputs =26 )$output
#values are too low for gpp in latter years

#using augmented data to predict gpp from r3pgn model
# Set the desired chunk size
chunk_size <- 72

# Get the total number of observations in the data
total_obs <- nrow(augmenteddataarray)

# Initialize an empty vector to store the results
output <- numeric(0)

# Loop through the data in chunks
for (i in seq(1, total_obs, by = chunk_size)) {
  # Extract the current chunk of data
  current_chunk <- augmenteddataarray[i:min(i + chunk_size - 1, total_obs), , , drop = FALSE]
  
  # Run the r3pgn model on the current chunk
  current_output <- r3pgn(siteData = datasiteaug, climate = current_chunk, thinning = NULL, parameters = data_param[, 2], outputs = 26)$output
  
  # Append the results to the output vector
  output <- c(output, current_output)
}

output1<-data.frame(output)

#saving predicted gpp for nas and training
caliaugggpnas<-output1[c(121:480),]
caliauggpptrain<-output1[c(481:1440),]


write.csv(caliaugggpnas, "augr3pgnNASpred.csv", row.names = FALSE)
write.csv(caliauggpptrain, "augr3pgntrainpred.csv", row.names = FALSE)
