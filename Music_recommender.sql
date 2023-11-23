Create table user_artists (
	userID int not null,
	artistID int not null,
	weight float not null
);

create table user_taggedartists(
	userID int not null,
	artistID int not null,
	tagID int not null);
	
create table tags (
	tagID int not null,
	tagValue varchar(200) not null);