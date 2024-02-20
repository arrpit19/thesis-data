
library(BayesianTools)


#new traindata data from hyytiala 2001:2006 used and same with gpp.
newtraindata1<-newtraindata
newtraindata1$gpp<-gppnew

  
  #  data_param was loaded and taken from r3pgn github. it has the default parameters of r3pgn. download the data_param.rda for this. it is in the data folder  
  #https://github.com/ForModLabUHel/threePGN-package/blob/master/r3pg/data/data_param.rda
  # select the parameters to be calibrated:
  pars2tune <- data_param 
  thispar <- data_param$mode
  names(thispar) <- data_param$parName
  
  
  
  CVfit <- matrix(NA, nrow=nrow(pars2tune), ncol = length(unique(newtraindata1$years)))
  
  i <- 1
  for (years in unique(newtraindata1$years)){
    
    df <- newtraindata1[newtraindata1$years != years,]
    
    ell <- function(pars2tune, data=df){
      # pars is a vector the same length as pars2tune
      thispar[pars2tune] <- pars2tune
      # likelihood function, first shot: normal density
      with(data, sum(dnorm(df$gpp, mean=r3pgn(siteData=datasite, climate=climatedata_arraytrain, thinning = NULL, parameters = data_param[,2], outputs =26 )$output, sd=thispar[32], log=T)))
    }
    priors <- createUniformPrior(lower=data_param$min, upper=data_param$max, best=data_param$mode)
    setup <- createBayesianSetup(likelihood=ell, prior=priors, parallel=T)
    settings <- list(iterations=50000, adapt=T, nrChains=3, parallel=T) # runs 3 chains in parallel for each chain ...
    # run:
    fit <- runMCMC(bayesianSetup = setup, settings = settings, sampler = "DREAMzs")
   
    
    pars_fit <- data_param
    pars_fit$mode <- MAP(fit)$parametersMAP
    CVfit[,i] <- pars_fit$mode
    i = i+1
  }
  
 
  write.csv(CVfit,file = "cv.csv", row.names = FALSE)
 
  
  #read the parameters cv file.   
  newparas<-read.csv("cv.csv")
  
  #for mean values
  newparasmean<-rowMeans(newparas)
  
  #using last three because training period was taken as 2004:2006. 2002 and 2003 for NAS)
  newparasvali<-newparas[,c(3:6)]
  
  #using mean of cv dataframe as the new parameter data. 
  #for training
  calimodel<-r3pgn(siteData=datasite, climate=climatedata_arraytrain, thinning = NULL, parameters = newparasmean, outputs =26 )
  
  #for test predictions
  calimodel2008<-r3pgn(siteData = datasite2008, climate = climatedata_array2008,thinning = NULL, parameters = newparasmean, outputs =26 )
  y2008pred<-calimodel2008[["output"]]
  
  write.csv(y2008pred, "ypred2008cali.csv", row.names = FALSE)
  
  #for training
  calimodelHY<-r3pgn(siteData =datasite , climate = climatedata_arraytrain,thinning = NULL, parameters = newparasmean, outputs =26 )
  yHYcali<-calimodelHY[["output"]]
  
  yh<-as.data.frame((yHYcali))
  
  #for NAS
  write.csv(yh[c(13:36),], "ypredNAS.csv", row.names = FALSE)
  
  #for training period
  write.csv(yh[c(37:72),],"ytrainpred.csv", row.names = FALSE)
  
  
  k<-read.csv("ypred2008cali.csv")
  
  #calculating mean abso error
  gpp_train <- matrix(NA, nrow=nrow(hyytiala_train), ncol=length(unique(newtraindata1$years)))
  gpp_test <- matrix(NA, nrow=12, ncol=3)
  
  
  maek<-matrix(NA, nrow=3,ncol = 1)
  rmse<-matrix(NA, nrow=3, ncol=1)
  
  
  #for test data, calculating mean absolute error using r3pgn parameters of 3 training periods.
  #gpptestnew is in hyytiala test script , which is gpp for the year 2008
  i <- 1
  for (i in 1:3){
    
    
    
    
    gpp_test[,i] <- r3pgn(siteData = datasite2008, climate = climatedata_array2008,thinning = NULL, parameters = newparasvali[,i], outputs =26 )$output
    maek[i,] <- sum(abs(gpptestnew - gpp_test[,i]))/length(gpp_test[,i])
    rmse[i,] <- sqrt(sum((gpptestnew - gpp_test[,i])^2)/length(gpp_test[,i]))
    
    i <- i+1
    
    
     }
 
   

  
#write.csv(maek, "maetemp.csv", row.names = FALSE)
