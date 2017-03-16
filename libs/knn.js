'use strict';

module.exports = function () {
    let app = this;

    let assert = (condition, message) => {
        if (!condition) {
            message = message || "Assertion failed";
            if (typeof Error !== "undefined") {
                throw new Error(message);
            }
            throw message;
        }
    };

    let _trained = null;

    app.setModel = (trained) => {
        assert(trained.constructor == Object, `dataset undefined`);
        _trained = trained;
    };

    app.getModel = () => _trained;

    let formatting = (row, label) => {
        if (!_trained) _trained = {};
        if (!_trained.index) _trained.index = 0;
        if (!_trained.featureBase) _trained.featureBase = {};
        if (!_trained.itemLabelBase) _trained.itemLabelBase = {};
        if (!_trained.itemBase) _trained.itemBase = {};

        _trained.index++;

        let id = _trained.index;

        _trained.itemBase[id] = {};
        _trained.itemLabelBase[id] = label;

        for (let key in row) {
            _trained.itemBase[id][key] = row[key] * 1;
            if (!_trained.featureBase[key]) _trained.featureBase[key] = [];
            _trained.featureBase[key].push(id);
        }
    };

    app.train = (dataset, labels) => {
        assert(dataset, `dataset undefined`);
        assert(labels, `labels undefined`);

        if (Array.isArray(dataset) === false) dataset = [dataset];
        else if (typeof dataset[0] != 'object') dataset = [dataset];

        if (Array.isArray(labels) === false) labels = [labels];

        assert(dataset.length === labels.length, `mismatched array length`);

        for (let i = 0; i < dataset.length; i++)
            formatting(dataset[i], labels[i]);

        return _trained;
    };

    let kNN = (item, k) => {
        if (!k) k = 3;
        let result = [];

        let relatedItems = {};

        for (let key in item)
            for (let i = 0; i < _trained.featureBase[key].length; i++)
                relatedItems[_trained.featureBase[key][i]] = {f: _trained.itemBase[_trained.featureBase[key][i]], label: _trained.itemLabelBase[_trained.featureBase[key][i]]};

        for (let itemId in relatedItems) {
            let comparison = relatedItems[itemId].f;
            let dist = 0;
            let keys = {};
            for (let key in comparison) keys[key] = true;
            for (let key in item) keys[key] = true;

            for (let j in keys)
                dist += ((comparison[j] ? comparison[j] * 1 : 0) - (item[j] ? item[j] * 1 : 0)) * ((comparison[j] ? comparison[j] * 1 : 0) - (item[j] ? item[j] * 1 : 0));
            dist = Math.sqrt(dist);
            result.push({label: _trained.itemLabelBase[itemId], dist: dist});
        }

        result.sort((a, b) => a.dist - b.dist);
        result.splice(k);

        let map = {};
        for (let i = 0; i < result.length; i++) {
            if (typeof map[result[i].label] === 'undefined') map[result[i].label] = {val: 0, cnt: 0};
            map[result[i].label].val += result[i].dist;
            map[result[i].label].cnt++;
        }

        let selected = null, min = null;
        for (let label in map) {
            map[label] = map[label].val / map[label].cnt;
            if (min === null) {
                selected = label;
                min = map[label];
            }

            if (map[label] < min) {
                selected = label;
                min = map[label];
            }
        }

        return selected;
    };

    app.test = (dataset, k, process) => {
        assert(dataset, `dataset undefined`);
        if (Array.isArray(dataset) === false) dataset = [dataset];
        else if (typeof dataset[0] != 'object') dataset = [dataset];

        let result = [];
        let st = new Date().getTime();
        for (let i = 0; i < dataset.length; i++) {
            result.push(kNN(dataset[i], k));
            if (process) {
                let pc = (new Date().getTime() - st) / 1000;
                process(i, i * 100 / dataset.length, pc)
            }
        }

        return result;
    };

    return app;
};