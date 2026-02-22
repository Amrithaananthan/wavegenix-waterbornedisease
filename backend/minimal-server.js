const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 5000;
const dataFile = path.join(__dirname, 'data.json');

// Initialize data file
if (!fs.existsSync(dataFile)) {
    fs.writeFileSync(dataFile, JSON.stringify([]));
}

// Simple in-memory data storage
let iotData = JSON.parse(fs.readFileSync(dataFile, 'utf8'));

// Generate sample data every 3 seconds
setInterval(() => {
  const newData = {
    deviceId: 'device_' + Math.floor(Math.random() * 100),
    pH: parseFloat((6.0 + Math.random() * 3).toFixed(2)),
    turbidity: parseFloat((Math.random() * 40).toFixed(2)),
    tds: Math.floor(50 + Math.random() * 1000),
    temperature: parseFloat((15 + Math.random() * 25).toFixed(2)),
    timestamp: new Date().toISOString(),
    location: { lat: 26.1445, lng: 91.7362 }
  };
  
  iotData.push(newData);
  if (iotData.length > 100) iotData = iotData.slice(-100);
  
  // Save to file
  fs.writeFileSync(dataFile, JSON.stringify(iotData, null, 2));
  
  console.log('Added data:', newData);
}, 3000);

// Create HTTP server
const server = http.createServer((req, res) => {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }
  
  if (req.url === '/' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ message: 'AquaGuard Mini Server Running!' }));
  }
  else if (req.url === '/api/data' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(iotData.slice(-50)));
  }
  else {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not found' }));
  }
});

server.listen(PORT, () => {
  console.log('Server running on port ' + PORT);
});
