'use strict';

const fs = require('fs');

module.exports = function () {
    let app = this;

    let trained = {};

    app.maxRelatedUser = 0;
    app.maxRelatedItem = 0;

    // dataset, user col, item col, value
    app.train = (data, x, y, val)=> {
        if (!x) x = 0;
        if (!y) y = 1;
        if (!val) val = 2;
        let userBase = {}, itemBase = {}, itemRank = {};

        for (let i = 0; i < data.length; i++) {
            let userId = data[i][x];
            let itemId = data[i][y];
            let feature = data[i][val] * 1;

            if (!userBase[userId]) userBase[userId] = [];
            userBase[userId].push({itemId: itemId, feature: feature});

            if (!itemBase[itemId]) itemBase[itemId] = [];
            itemBase[itemId].push({userId: userId, feature: feature});

            if (!itemRank[itemId]) itemRank[itemId] = 0;
            itemRank[itemId] += feature;
        }

        let ranking = [];
        for (let itemId in itemRank)
            ranking.push({itemId: itemId, play: itemRank[itemId]});
        ranking.sort((a, b)=> b.play - a.play);

        trained.userBase = userBase;
        trained.itemBase = itemBase;
        trained.ranking = ranking;
    };

    app.getModel = ()=> trained;

    app.setModel = (model)=> trained = model;

    app.recommendByArray = (listenList, count)=> {
        let alreadyIn = {};
        let similarUsers = {};
        for (let i = 0; i < listenList.length; i++) {
            alreadyIn[listenList[i].itemId] = true;
            let similarUserList = trained.itemBase[listenList[i].itemId];
            for (let j = 0; j < similarUserList.length; j++) {
                if (!similarUsers[similarUserList[j].userId])
                    similarUsers[similarUserList[j].userId] = 0;
                similarUsers[similarUserList[j].userId] += similarUserList[j].feature * listenList[i].feature;
            }
        }

        let relatedUsers = [];
        for (let userId in similarUsers)
            relatedUsers.push({id: userId, score: similarUsers[userId]});
        relatedUsers.sort((a, b)=> b.score - a.score);

        let playlist = {};
        let playlistCount = 0;
        for (let i = 0; i < relatedUsers.length; i++) {
            if (app.maxRelatedUser !== 0 && i > app.maxRelatedUser)
                break;
            let userId = relatedUsers[i].id;
            let userScore = relatedUsers[i].score;
            let userList = trained.userBase[userId];

            for (let j = 0; j < userList.length; j++) {
                if (alreadyIn[userList[j].itemId]) continue;
                if (app.maxRelatedItem !== 0 && j > app.maxRelatedItem)
                    break;
                if (!playlist[userList[j].itemId]) {
                    playlist[userList[j].itemId] = 0;
                    playlistCount++;
                }

                playlist[userList[j].itemId] += userScore;
            }
        }

        let result = [];
        for (let itemId in playlist)
            result.push({itemId: itemId, score: Math.round(Math.log(playlist[itemId] + 1) * 100) / 100});
        result.sort((a, b)=> b.score - a.score);
        result.splice(count);

        for (let i = 0; i < trained.ranking.length; i++) {
            if (result.length >= count) break;
            if (!playlist[trained.ranking[i].itemId])
                result.push(trained.ranking[i]);
        }

        return result;
    };

    app.recommendToUser = (userId, count) => {
        let userList = trained.userBase[userId];
        if (userList)
            return app.recommendByArray(userList, count);
        else
            return JSON.parse(JSON.stringify(trained.ranking)).splice(0, count);
    };

    app.recommendToUsers = (userIds, count, process) => {
        let result = {};
        for (let i = 0; i < userIds.length; i++) {
            result[userIds[i]] = app.recommendToUser(userIds[i], count);
            if (process) process(i + 1);
        }

        return result;
    };

    app.recommendGT = (gt, count, process) => {
        let result = {}, cnt = 0;
        for (let userId in gt) {
            result[userId] = app.recommendToUser(userId, count);
            if (process)  process(++cnt);
        }

        return result;
    };

    app.gt = (data, x, y, val)=> {
        if (!x) x = 0;
        if (!y) y = 1;
        if (!val) val = 2;

        let userBase = {};
        for (let i = 0; i < data.length; i++) {
            let userId = data[i][x];
            let itemId = data[i][y];
            let feature = data[i][val] * 1;

            if (!userBase[userId]) userBase[userId] = [];
            userBase[userId].push({itemId: itemId, feature: feature});
        }

        return userBase;
    };

    return app;
};