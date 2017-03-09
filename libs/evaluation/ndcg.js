'use strict';

module.exports = (gt, recs)=> {
    let Q = 0.0, S = 0.0;

    for (let u in gt) {
        if (!recs[u])
            continue;

        let vs = gt[u];
        let rec = recs[u];
        if (!rec) continue;

        let idcg = 0;
        let vsContain = {};
        for (let i = 0; i < vs.length; i++) {
            idcg += 1.0 / Math.log(i + 2, 2);
            vsContain[vs[i].itemId] = true;
        }

        let dcg = 0.0;
        for (let i = 0; i < rec.length; i++) {
            let r = rec[i].itemId;
            if (!vsContain[r]) continue;
            let rank = i + 1;
            dcg += 1.0 / Math.log(rank + 1, 2);
        }

        let ndcg = dcg / idcg;
        S += ndcg;
        Q += 1;
    }

    return S / Q;
};