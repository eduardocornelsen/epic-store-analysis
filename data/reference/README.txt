# Authors: Samuel de Souza Gomes, Alexandre Magno Sousa
# Title: Epic Games Store Dataset
# Year: 2022

Entity Relationship Model:

- Firstly, the "ERM.png" file contains the image for the entity relationship model.

The next section describes each one of tables present in dataset and ERM file.

TABLES:

1. Games table ("games.csv"):

Description of games file columns:

- ID: 			identification of game.
- Name: 		name of game.
- Game slug: 		short name of game.
- Price: 		price of game.
- Release date:		release date of game.
- Platform:		platforms that the game is available.
- Description:		description of game.
- Developer:		company that developed the game.
- Publisher:		company that published the game.
- Genres		genres of game

2. Necessary Hardware table ("necessary_hardware.csv"):

All the games has a minimum and recommended necessary hardware.

Description of necessary hardware file columns:

- ID: 			identification of necessary hardware.
- Operational system:	operational system the game supports.
- Processor:		processor minimum or recommended that game needs to run.
- Memory:		RAM memory minimum or recommended that game needs to run.
- Graphics:		graphics minimum or recommended that game needs to run.
- Game ID:		identification of game.

3. Social networks table ("social_networks.csv"):

Description of social networks file columns:

- ID: 			identification of social network.
- Description: 		name of social network (such as "linkTwitter").
- URL			URL of social network of the game.
- Game ID:		identification of game.

4. Twitter accounts table ("twitter_accounts.csv"):

Descriptin of twitter accounts file columns:

- ID: 			anonymized identification of twitter account.
- Name:			profile name of twitter account.
- Username:		username of twitter account.
- Bio:			biography of twitter account, such as a little description.
- Location		location of twitter account owner.
- Website		in case that user has a website, contain the link.
- Joined date		date that user created your twitter account.
- Following		quantity of users that profile is following.
- Followers		quantity of users that follow the profile.
- Game ID		identification of game.

5. Tweets table ("tweets.csv"):

Description of tweets file columns:

- ID:			anonymized identification of tweet.
- Text:			text of tweet.
- URL Media:		image or video link of tweet.
- Quantity likes:	quantity likes of tweet.
- Quantity retweets:	quantity retweets of tweet.
- Quantity quotes:	quantity quotes of tweet.
- Quantity replys:	quantity replys of tweet.
- In Reply To User ID:	user id that posted parent tweet.
- Timestamp:		date and hour that tweet was posted.
- Twitter Account ID:	anonymized identification of twitter account.

6. Open Critic table ("open_critic.csv")

Description of open critic file columns:

- ID:			identification of critic.
- Rating:		rating of game (example: 80).
- Comment:		author comment about the game.
- Company:		company name that rated the game.
- Author:		author name that rated the game.
- Date:			date of critic.
- Description:		description of game.
- Top critic:		verify if is a top critic (authors with verdict).
- Game ID:		identification of game.