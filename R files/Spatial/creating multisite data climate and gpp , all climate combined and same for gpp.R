
#combining multi site data for climate and 
#multisitedata<- cbind(monthly_meanstrainclimate[c(37:48),], gppnew[c(37:48)])
#multisite2<-cbind(climatedatakriz[], gppkriz)
#multisite3<-cbind(climatedatacole[], gppcole2004)
#multisite4<-cbind(climatedatasoro[], gppsoro)


#creating multi site combined climate
multisiteclim2008<- rbind(
                                                   climatedatakriz2008 ,
                                                    climatedatacole2008 ,
                                                   climatedatasoro2008)

#scaling the data
multisiteclim2008S<-scale(multisiteclim2008)

#multi site data with both climate and site data combined?!
# sites together and hyytiala separetely because that is the unseen site to be tested
multi3sites2008<-rbind(kriztestmulti2008,coletestmulti2008,sorotestmulti2008)
multisite2008<-multi3sites2008[,-8]
multi3sites2008S<-scale(multisite2008)

multisiteHY2008<-hytestmulti2008
multisiteHY2008S<-multisiteHY2008

testHY2008<-multisiteHY2008S[,-8]
testHY2008S<-scale(testHY2008)

#gpp 
multisitegpp2008<-cbind(gppkriz2008,gppcole2008,gppsoro2008)
j2008<-data.frame(multisitegpp2008)
multisitegppk2008<-stack(j2008)

multi4sitegpp2008<-cbind(gpptestnew, gppkriz2008,gppcole2008,gppsoro2008)
j42008<-data.frame(multi4sitegpp2008)
multi4sitegppk2008<-stack(j42008)


#2008 3 site climate
write.csv(multisiteclim2008, "multisiteclim2008.csv", row.names = FALSE)
write.csv(multisiteclim2008S, "multisiteclimS2008.csv", row.names = FALSE)

#2008 3 site climate+site combined data
write.csv(multisite2008, "multi3siteclim2008.csv", row.names = FALSE)
write.csv(multi3sites2008S, "multi3siteclimS2008S.csv", row.names = FALSE)


write.csv(testHY2008, "hy2008multisite.csv", row.names = FALSE)
write.csv(testHY2008S, "hy2008multisiteS.csv", row.names = FALSE)







write.csv(multisitegppk2008[,1], "multi3sitegpp2008.csv", row.names = FALSE)


write.csv(multi4sitegppk2008[,1], "multi4sitegpp2008.csv", row.names = FALSE)

write.csv(gpptestnew, "multisitetestHy.csv", row.names = FALSE)









j<-data.frame(multisitegpp)
