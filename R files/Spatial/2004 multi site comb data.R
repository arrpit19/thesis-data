#multisitedata<- cbind(monthly_meanstrainclimate[c(37:48),], gppnew[c(37:48)])
#multisite2<-cbind(climatedatakriz[], gppkriz)
#multisite3<-cbind(climatedatacole[], gppcole2004)
#multisite4<-cbind(climatedatasoro[], gppsoro)

#combining data for 2004 
multisiteclim<- rbind(hytestmulti2004,kriztestmulti2004,coletestmulti2004,sorotestmulti2004)

multisiteclimS<-scale(multisiteclim)

multisitegpp<-cbind(gppnew[c(37:48)],gppkriz,gppcole2004,gppsoro)

multisitegppk<-stack(j)



write.csv(multisiteclim, "multisiteclim2004.csv", row.names = FALSE)
write.csv(multisiteclimS[,-8], "multisiteclimS2004.csv", row.names = FALSE)


write.csv(multisitegppk[,1], "multisitegpp2004.csv", row.names = FALSE)
write.csv(multisitegppk[c(13:48),1], "multisitegpp3site.csv", row.names = FALSE)






j<-data.frame(multisitegpp)
