# Crime-Project-Pipeline
Overview: Pipeline creating reports on counties' crime levels and property information 

## Datasets used 
+ The first dataset I used can be downloaded at https://data.police.uk/data/ . This works by selecting the police force and timeframe you want data on before you download it. In the case of my project I downloaded 2 years of data from 2022-05 to 2024-04 on the police forces: Essex Police, Hertfordshire Constabulary, Kent Police and Surrey Police as I wanted to focus on the home counties around London specifically. I also only ticked 'Include crime data' and none of the other options. This gives you a folder with 96 csv's which I then saved as 'Police crime dataset'.
+ The second dataset I used can be downloaded at https://www.freemaptools.com/download-uk-postcode-lat-lng.htm . Each row details of the longitude and latitude of postcodes all around the UK. Since the 'Police crime dataset' has the longitude and latitude of each crime I wanted to use this second dataset to get a postcode of each crime for later analysis.
+ The third and fourth dataset I used can be downloaded https://landregistry.data.gov.uk/app/ukhpi/browse?from=2022-05-01&location=http%3A%2F%2Flandregistry.data.gov.uk%2Fid%2Fregion%2Fessex&to=2024-04-01&lang=en . After completing a previous project I identified that the number burglaries + robberies per km^2 was less in Essex and Kent so I picked these two counties as the ones I would get property datasets on. At this link you can download two separate csv's for these counties detailiing how property prices and sales change each month. I did this from 2022-5 to 2024-04 to match my 'Police crime dataset'

## 
