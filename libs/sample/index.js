'use strict';

((app) => {
    const fs = require('fs');
    const path = require('path');

    app.movie = () => {
        let dataset = fs.readFileSync(path.resolve(__dirname, 'res', 'movie.txt'), 'utf-8').split('\n');
        let labels = [];
        for (let i = 0; i < dataset.length; i++) {
            if (!dataset[i] || dataset[i].length == 0) {
                dataset.splice(i, 1);
                continue;
            }

            dataset[i] = dataset[i].split(' ');
            let [movie_id, user_id, rating, like] = dataset[i];
            dataset[i] = {movie_id: movie_id, user_id: user_id, rating: rating, like: like};
        }

        return dataset;
    };

    app.iris = () => {
        let dataset = fs.readFileSync(path.resolve(__dirname, 'res', 'iris.data'), 'utf-8').split('\n');
        let labels = [];
        for (let i = 0; i < dataset.length; i++) {
            if (!dataset[i] || dataset[i].length == 0) {
                dataset.splice(i, 1);
                continue;
            }

            dataset[i] = dataset[i].split(',');
            labels[i] = dataset[i].splice(dataset[i].length - 1, 1)[0];
        }

        return {dataset: dataset, labels: labels};
    };

    app.yeast = () => {
        let dataset = fs.readFileSync(path.resolve(__dirname, 'res', 'yeast.txt'), 'utf-8').split('\n');
        let labels = [];
        for (let i = 0; i < dataset.length; i++) {
            if (!dataset[i] || dataset[i].length == 0) {
                dataset.splice(i, 1);
                continue;
            }

            dataset[i] = dataset[i].split('  ');
            dataset[i].splice(0, 1);
            labels[i] = dataset[i].splice(dataset[i].length - 1, 1)[0];
        }

        return {dataset: dataset, labels: labels};
    };

    app.bbc = () => {
        let mtx = fs.readFileSync(path.resolve(__dirname, 'res', 'bbc.mtx'), 'utf-8').split('\n');
        let classes = fs.readFileSync(path.resolve(__dirname, 'res', 'bbc.classes'), 'utf-8').split('\n');

        let dataset = [];
        let labels = [];
        for (let i = 0; i < classes.length; i++) {
            if (!classes[i] || classes[i].length == 0) continue;
            labels.push(classes[i].split(' ')[1]);
        }

        for (let i = 0; i < mtx.length; i++) {
            if (!mtx[i] || mtx[i].length == 0) continue;
            mtx[i] = mtx[i].split(' ');
            if (!dataset[mtx[i][1] * 1 - 1]) dataset[mtx[i][1] * 1 - 1] = {};
            dataset[mtx[i][1] * 1 - 1][mtx[i][0]] = mtx[i][2] * 1;
        }

        return {dataset: dataset, labels: labels};
    };
})(module.exports);