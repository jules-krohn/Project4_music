CREATE TABLE users (
	userID SERIAL PRIMARY KEY
);
CREATE TABLE artists (
    artistID SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL
);
CREATE TABLE tags (
    tagID SERIAL PRIMARY KEY,
    tagValue VARCHAR(200) NOT NULL
);
CREATE TABLE user_artists (
    userID INT NOT NULL,
    artistID INT NOT NULL,
    weight INT NOT NULL,
    FOREIGN KEY (userID) REFERENCES users (userID),
    FOREIGN KEY (artistID) REFERENCES artists(artistID)
);
CREATE TABLE user_taggedartists (
    userID INT NOT NULL,
    artistID INT NOT NULL,
    tagID INT NOT NULL,
    day INT,
    month INT,
    year INT,
    FOREIGN KEY (userID) REFERENCES users (userID),
    FOREIGN KEY (tagID) REFERENCES tags(tagID) 
);

--ALTER TABLE user_taggedartists DROP CONSTRAINT user_taggedartists_artistid_fkey;

-- Disable foreign key checks
--BEGIN;
--SET CONSTRAINTS ALL DEFERRED;
--COMMIT;

-- Re-enable
--BEGIN;
--SET CONSTRAINTS ALL IMMEDIATE;
--COMMIT;

