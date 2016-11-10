'use strict';

((app)=> {
    const fs = require('fs');
    const path = require('path');

    app.yeast = ()=> new Promise((resolve)=> {
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

        resolve({dataset: dataset, labels: labels});
    });

    app.bbc = ()=> new Promise((resolve)=> {
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

        resolve({dataset: dataset, labels: labels});
    });

    app.acoustic_scale = () => new Promise((resolve)=> {
        let dataset = fs.readFileSync(path.resolve(__dirname, 'res', 'acoustic_scale.t'), 'utf-8').split('\n');
        let labels = [];
        for (let i = 0; i < dataset.length; i++) {
            if (!dataset[i] || dataset[i].length == 0) {
                dataset.splice(i, 1);
                continue;
            }

            dataset[i] = dataset[i].split(' ');
            labels[i] = dataset[i].splice(0, 1)[0] * 1;

            let reformat = {};
            for (let j = 0; j < dataset[i].length; j++) {
                if (!dataset[i][j] || dataset[i][j].length == 0) {
                    dataset[i].splice(j, 1);
                    continue;
                }

                dataset[i][j] = dataset[i][j].split(':');
                reformat[dataset[i][j][0]] = dataset[i][j][1] * 1;
            }

            dataset[i] = reformat;
        }

        resolve({dataset: dataset, labels: labels});
    });
})(module.exports);