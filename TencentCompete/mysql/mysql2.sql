use tencent;

show table status;
select * from train where label=1;

select * from ad;
select count(*) from app_categories where appCategory=0;

select a.*, b.appCategory from ad a left join app_categories b on a.appID=b.appID;

create table origin_ad select ad.*, app_categories.appCategory from ad left join app_categories on ad.appID=app_categories.appID;

select count(*) from origin_ad where appCategory<100;

select count(*) from ad where appID<=472;
select * from app_categories where appID<=472;

select count(*) from origin_ad;

select * from users;


