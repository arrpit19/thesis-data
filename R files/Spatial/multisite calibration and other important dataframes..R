#creating spatial data common dataframe with all 4 sites. 
library(dplyr)

#for 2004 NAS 
newsitedata2004<-rbind(datasite2004,datasitekriz,datasitecole,datasitesoro)


newsitedata2008<-rbind(datasite2008,datasitekriz2008,datasitecole2008,datasitesoro2008)


# Create a column for dataframe IDs. this is done so that the combined climate array for r3pgn input can have Climid which is important so recignize which site's climate data it is.  

#combined_df2008ll<-rbind(climatedata2008 ,
                 #        climatedatakriz2008 ,
                  #       climatedatacole2008 ,
                   #      climatedatasoro2008)

#monthly_meanstrainclimate is hyytiala and c(37:48) is for the year 2004
combined_df2004ll<-rbind(monthly_meanstrainclimate[c(37:48),] ,
                        climatedatakriz ,
                         climatedatacole ,
                         climatedatasoro)






#combined_df2008 <- bind_rows(climatedata2008 %>% mutate(dataframe_id = 1),
            #             climatedatakriz2008 %>% mutate(dataframe_id = 2),
             #            climatedatacole2008 %>% mutate(dataframe_id = 3),
              #           climatedatasoro2008 %>% mutate(dataframe_id = 4))

#2008 multisite climnate data 3-D array 
#climatedata_array2008comb <- array(data = matrix(unlist(combined_df2008ll), nrow = nrow(climatedata2008), ncol = ncol(climatedata2008)), dim = c(nrow(climatedata2008), ncol(climatedata2008), 4))



# this is done so that the combined climate array for r3pgn input can have Climid which is important so recignize which site's climate data it is.  

# Create a list of data frames this is for the year 2008 
dfs <- list(climatedata2008 ,
            climatedatakriz2008 ,
            climatedatacole2008 ,
            climatedatasoro2008)

# Determine the dimensions of the resulting 3D array
num_dataframes <- length(dfs)
max_rows <- max(sapply(dfs, nrow))
max_cols <- max(sapply(dfs, ncol))

# Initialize an empty 3D array
result_array2008 <- array(NA, dim = c(max_rows, max_cols, num_dataframes), dimnames = list(NULL, NULL, NULL))

# Fill the 3D array with data frames
for (i in 1:num_dataframes) {
  result_array2008[1:nrow(dfs[[i]]), 1:ncol(dfs[[i]]), i] <- as.matrix(dfs[[i]])
}


#testing = result_array2008[,,3]


#trying the model with the inputs
trymulti<-r3pgn(siteData=newsitedata2008, climate=result_array2008, thinning = NULL, parameters = data_param[,2], outputs =26 )

# climate data for year 2004 of hyytiala,   
climatehy2004<- monthly_meanstrainclimate[c(37:48),]



# Create a list of  data frames, this is for the year 2004, to be use for NAS
dfs2004 <- list(monthly_meanstrainclimate[c(37:48),],
                 climatedatakriz ,
                 climatedatacole ,
                 climatedatasoro)

                                
# Determine the dimensions of the resulting 3D array
num_dataframes2004 <- length(dfs2004)
max_rows2004 <- max(sapply(dfs2004, nrow))
max_cols2004 <- max(sapply(dfs2004, ncol))

# Initialize an empty 3D array
result_array2008 <- array(NA, dim = c(max_rows, max_cols, num_dataframes), dimnames = list(NULL, NULL, NULL))
result_array2004 <- array(NA, dim = c(max_rows, max_cols, num_dataframes), dimnames = list(NULL, NULL, NULL))




# Fill the 3D array with data frames
for (i in 1:num_dataframes) {
  result_array2008[1:nrow(dfs[[i]]), 1:ncol(dfs[[i]]), i] <- as.matrix(dfs[[i]])
}

# Fill the 3D array with data frames
for (i in 1:num_dataframes) {
  result_array2004[1:nrow(dfs[[i]]), 1:ncol(dfs2004[[i]]), i] <- as.matrix(dfs2004[[i]])
}


#now for gpp  

#dfsgpp2004 <- rbind(gppnew[c(37:48)],

              #  (gppkriz),
               # (gppcole2004),
                #(gppsoro))


#gpp2004df<-t(dfsgpp2004)

# Determine the dimensions of the resulting 3D array
#num_dataframesg <- length(dfsgpp2004)
#max_rowsg <- 12
#max_colsg <- 1



#newtraindata$months <- rep(1:12, times = 6)


# calibration of r3pgn model for multiple sites.

#creating the gpp for 2004 for all 4 sites
cdfsgpp2004 <- cbind(gppnew[c(37:48)],
                    
                    (gppkriz),
                    (gppcole2004),
                    (gppsoro))

#converting it into a data frame
long_df <- as.data.frame(matrix(unlist(cdfsgpp2004), ncol = 1))

#combining the climate data and gpp data of all 4 sites. 
newdatacali2004<-cbind(combined_df2004ll,long_df)

#creating cv fit matrix
CVfitmulti <- matrix(NA, nrow=nrow(pars2tune), ncol = 4)


#now doiing calibration with cdfsgpp2004 which has all gpp data and newdatacali2004 which has climate and gpp data. 
#site data is newsitedata2004 that was created above and result_array2004 as climate 3 D array 
i <- 1

for (i in 1:4){
  
  
  
  ell <- function(pars2tune, data=newdatacali2004){
    # pars is a vector the same length as pars2tune
    thispar[pars2tune] <- pars2tune
    # likelihood function, first shot: normal density
    with(data, sum(dnorm(cdfsgpp2004[,-i], mean=r3pgn(siteData=newsitedata2004[-i,], climate=result_array2004[,,-i], thinning = NULL, parameters = data_param[,2], outputs =26 )$output, sd=thispar[32], log=T)))
  }
  priors <- createUniformPrior(lower=data_param$min, upper=data_param$max, best=data_param$mode)
  setup <- createBayesianSetup(likelihood=ell, prior=priors, parallel=T)
  settings <- list(iterations=50000, adapt=T, nrChains=3, parallel=T) # runs 3 chains in parallel for each chain ...
  # run:
  fit <- runMCMC(bayesianSetup = setup, settings = settings, sampler = "DREAMzs")

  pars_fit <- data_param
  pars_fit$mode <- MAP(fit)$parametersMAP
  CVfitmulti[,i] <- pars_fit$mode
  i = i+1
}

#taking the mean of CVfit parameters
newparamulti<-rowMeans(CVfitmulti)

#gpp predictions now with calibrated model. 
gpp_testmulti<- matrix(NA, nrow=12, ncol=4)
#mean abso error
maekmulti<-matrix(NA, nrow=4,ncol = 1)

#predicted gpp by r3pgn for all sites  for test data
gpp_testmulti[] <- r3pgn(siteData = newsitedata2008, climate = result_array2008,thinning = NULL, parameters = newparamulti, outputs =26 )$output


#mae error for all sites.
i<-1

for (i in 1:4){
  
  
  
  
    maekmulti[i,] <- sum(abs(cdfsgpp2004[,i] - gpp_testmulti[,i]))/length(gpp_testmulti[,i])
  
  
  i <- i+1
  
  
}

#preedicted gpp for 2004 multi site data
gpptrainmulti2004<- matrix(NA, nrow=12, ncol=4)
gpptrainmulti2004[]<-r3pgn(siteData = newsitedata2004, climate = result_array2004,thinning = NULL, parameters = newparamulti, outputs =26 )$output

long_dfmulti <- as.data.frame(matrix(unlist(gpptrainmulti2004), ncol = 1))
write.csv(long_dfmulti, "gpptrainmulti2004.csv", row.names = FALSE)

#predicted gpp 2008 to be saved 
gpptrainmulti2008<- matrix(NA, nrow=12, ncol=4)
gpptrainmulti2008[]<-r3pgn(siteData = newsitedata2008, climate = result_array2008,thinning = NULL, parameters = newparamulti, outputs =26 )$output

long_dfmulti <- as.data.frame(matrix(unlist(gpptrainmulti2008), ncol = 1))
#saving kriz, colellongo and soro sites
write.csv(long_dfmulti[-c(1:12),], "gpptestmulti2008.csv", row.names = FALSE)

#saving hyytiala separately because this is the unseen site to be tested so saving seaparately. 
write.csv(long_dfmulti[c(1:12),], "gpptestmulti2008HY.csv", row.names = FALSE)
