##项目介绍##
本项目源于腾讯社交广告高校算法大赛，赛事网站[http://algo.tpai.qq.com/home/home/index.html](http://algo.tpai.qq.com/home/home/index.html "腾讯高校算法大赛")


### 初赛数据 ###
ad.csv总记录数——6582

|Table ad.csv | min | max |
| :--: | :--: | :--: |
|creativeID	| 1 | 6582 |
|adID	| 1 | 3616 |
|camgaignID	| 1 | 720 |
|advertiserID	| 1 | 91 |
|appID	| 14 | 472 |
|appPlatform	| 1 | 2 |

position.csv总记录数——7645

|Table position.csv | min | max |
| :--: | :--: | :--: |
|positionID	| 1 | 7645 |
|sitesetID	| 0 | 2 |
|positionType	| 0 | 5 |

app_categories.csv总记录数——217041

|Table app_categories.csv | min | max |
| :--: | :--: | :--: |
|appID	| 14 | 433269 |
|appCategory	| 0 | 503 |

user.csv总记录数——2805118

|Table user.csv | min | max |
| :--: | :--: | :--: |
|userID	| 1 | 2805118 |
|age	| 0 | 80 |
|gender	| 0 | 2 |
|education	| 0 | 7 |
|marriageStatus	| 0 | 3 |
|haveBaby	| 0 | 6 |
|hometown	| 0 | 3401 |
|residence	| 0 | 3401 |

user_app_actions.csv总记录数——6003471

|Table user_app_actions.csv | min | max |
| :--: | :--: | :--: |
|userID	| 1 | 2805118 |
|installTime	| 10000 | 302359 |
|appID	| 354 | 433267 |

user_installedapps.csv总记录数——84039009

|Table user_installedapps.csv | min | max |
| :--: | :--: | :--: |
|userID	| 1 | 2805117 |
|appID	| 354 | 433269 |

train.csv总记录数——3749528

|Table train.csv | min | max |
| :--: | :--: | :--: |
|label	| 0 | 1 |
|clickTime	| 170000 | 302359 |
|conversionTime	| 170005 | 302359 |
|creativeID	| 1 | 	6582 |
|userID	| 1 | 	2805118 |
|positionID	| 1 | 7645 |
|connectionType	| 0 | 4 |
|telecomsOperator	| 0 | 3 |

test.csv总记录数——338489

|Table test.csv | min | max |
| :--: | :--: | :--: |
|instanceID	| 1 | 338489 |
|label	| -1 | -1 |
|clickTime	| 310000 | 312359 |
|creativeID	| 4 | 6580 |
|userID	| 3 | 2805117 |
|positionID	| 2 | 7645 |
|connectionType	| 0 | 4 |
|telecomsOperator	| 0 | 3 |


|Table Name | length |
| :--: | :--: |
|ad	| 6582 |
|positions	| 7645 |
|app_categories	| 216900 |
|users	| 2799494 |
|user_app_actions	| 5989571 |
|user_installedapps	| 83855023 |
|train	| 3738773 |
|test	| 337931 |





