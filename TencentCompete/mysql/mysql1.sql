show databases;
#create database tencent;
use tencent;

show tables;

create table ad(
	creativeID smallint,
	adID smallint,
	camgaignID smallint,
	advertiserID tinyint,
	appID int,
	appPlatform tinyint,
    primary key(creativeID)
);

create table positions(
	positionID smallint,
    sitesetID tinyint,
    positionType tinyint,
    primary key(positionID)
);

create table app_categories(
	appID int,
    appCategory smallint
);

create table users(
	userID int,
    age tinyint,
    gender tinyint,
    education tinyint,
    marriageStatus tinyint,
    haveBaby tinyint,
    hometown smallint,
    residence smallint,
    primary key(userID)
);

create table user_app_actions(
	userID int,
    installTime int,
    appID int
);

create table user_installedapps(
	userID int,
    appID int
);

create table train(
	label tinyint,
    clickTime int,
    conversionTime int,
    creativeID smallint,
	userID int,
    positionID smallint,
    connectionType tinyint,
	telecomsOperator tinyint
);

create table test(
	instanceID int,
	label tinyint,
    clickTime int,
    creativeID smallint,
	userID int,
    positionID smallint,
    connectionType tinyint,
	telecomsOperator tinyint
);

show tables;
select * from ad;
select * from positions;

LOAD DATA LOCAL INFILE 'D:\\Codes\\Data\\TecentCompete\\pre\\ad.csv' INTO TABLE ad fields terminated by ',' lines terminated by '\n' ignore 1 rows;

LOAD DATA LOCAL INFILE 'D:\\Codes\\Data\\TecentCompete\\pre\\position.csv' INTO TABLE positions fields terminated by ',' lines terminated by '\n' ignore 1 rows;

LOAD DATA LOCAL INFILE 'D:\\Codes\\Data\\TecentCompete\\pre\\app_categories.csv' INTO TABLE app_categories fields terminated by ',' lines terminated by '\n' ignore 1 rows;

LOAD DATA LOCAL INFILE 'D:\\Codes\\Data\\TecentCompete\\pre\\user.csv' INTO TABLE users fields terminated by ',' lines terminated by '\n' ignore 1 rows;

LOAD DATA LOCAL INFILE 'D:\\Codes\\Data\\TecentCompete\\pre\\user_app_actions.csv' INTO TABLE user_app_actions fields terminated by ',' lines terminated by '\n' ignore 1 rows;

LOAD DATA LOCAL INFILE 'D:\\Codes\\Data\\TecentCompete\\pre\\user_installedapps.csv' INTO TABLE user_installedapps fields terminated by ',' lines terminated by '\n' ignore 1 rows;

LOAD DATA LOCAL INFILE 'D:\\Codes\\Data\\TecentCompete\\pre\\train.csv' INTO TABLE train fields terminated by ',' lines terminated by '\n' ignore 1 rows;

LOAD DATA LOCAL INFILE 'D:\\Codes\\Data\\TecentCompete\\pre\\test.csv' INTO TABLE test fields terminated by ',' lines terminated by '\n' ignore 1 rows;

select * from train;

#drop table users;


