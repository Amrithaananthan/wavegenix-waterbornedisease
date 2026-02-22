const fs = require('fs');
const path = require('path');

const dataFile = path.join(__dirname, 'data.json');
const alertsFile = path.join(__dirname, 'alerts.json');

// Initialize files if they don't exist
if (!fs.existsSync(dataFile)) {
    fs.writeFileSync(dataFile, JSON.stringify([]));
}
if (!fs.existsSync(alertsFile)) {
    fs.writeFileSync(alertsFile, JSON.stringify([]));
}

function readData() {
    try {
        return JSON.parse(fs.readFileSync(dataFile, 'utf8'));
    } catch (error) {
        return [];
    }
}

function writeData(data) {
    try {
        fs.writeFileSync(dataFile, JSON.stringify(data, null, 2));
        return true;
    } catch (error) {
        return false;
    }
}

function readAlerts() {
    try {
        return JSON.parse(fs.readFileSync(alertsFile, 'utf8'));
    } catch (error) {
        return [];
    }
}

function writeAlerts(alerts) {
    try {
        fs.writeFileSync(alertsFile, JSON.stringify(alerts, null, 2));
        return true;
    } catch (error) {
        return false;
    }
}

function addIOTData(newData) {
    let data = readData();
    newData.timestamp = new Date();
    newData._id = Date.now().toString();
    data.push(newData);
    
    if (data.length > 100) {
        data = data.slice(-100);
    }
    
    writeData(data);
    return newData;
}

function getAllData() {
    return readData();
}

function getRecentData(limit = 50) {
    const data = readData();
    return data.slice(-limit);
}

function addAlert(alert) {
    let alerts = readAlerts();
    alert.timestamp = new Date();
    alert._id = Date.now().toString();
    alerts.unshift(alert); // Add to beginning
    
    if (alerts.length > 20) {
        alerts = alerts.slice(0, 20); // Keep only recent 20 alerts
    }
    
    writeAlerts(alerts);
    return alert;
}

function getAlerts() {
    return readAlerts();
}

module.exports = {
    addIOTData,
    getAllData,
    getRecentData,
    addAlert,
    getAlerts
};
