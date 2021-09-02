require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('../load-csv')

const { features, labels, testFeatures, testLabels } = loadCSV('C:\\Users\\Anton\\MLCourse\\MLKits\\regressions\\logistic-regression\\cars.csv', {
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['passedemissions'],
    shuffle: true,
    splitTest: 50,
    converters: {
        passedemissions: (value) => {
           return value === 'TRUE' ? 1 : 0
        }
    }
})
console.log(labels)