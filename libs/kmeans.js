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

    app.train = (dataset, labels) => {
        assert(dataset, `dataset undefined`);
        assert(labels, `labels undefined`);

        if (Array.isArray(dataset) === false) dataset = [dataset];
        else if (typeof dataset[0] != 'object') dataset = [dataset];

        if (Array.isArray(labels) === false) labels = [labels];

        assert(dataset.length === labels.length, `mismatched array length`);
    };

    app.test = (dataset, k, process) => {
        assert(dataset, `dataset undefined`);
        if (Array.isArray(dataset) === false) dataset = [dataset];
        else if (typeof dataset[0] != 'object') dataset = [dataset];
    };

    return app;
};