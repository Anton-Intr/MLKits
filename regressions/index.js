require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('./load-csv')
const LinearRegression = require('./linear_regression')

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower'],
    labelColumns: ['mpg']
})

const regression = new LinearRegression(features, labels, {
    learningRate: 0.0001,
    iterations: 100
})
regression.train()
const r2 = regression.test(testFeatures, testLabels)
console.log('R2 is', r2)
// console.log('Updated M is', regression.weigths.get(1, 0))
// console.log('Updated B is', regression.weigths.get(0, 0))