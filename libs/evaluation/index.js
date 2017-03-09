'use strict';

module.exports = (()=> {
    let app = {};

    app.accuracy = require('./accuracy');
    app.ndcg = require('./ndcg');

    return app;
})();