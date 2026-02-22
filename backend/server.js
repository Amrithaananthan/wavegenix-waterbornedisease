const express = require('express');
const aiRoutes = require('./ai_routes');
const cors = require('cors');
const http = require('http');
const { Server } = require('socket.io');
const { addIOTData, getRecentData, addAlert, getAlerts } = require('./database');

const app = express();
const server = http.createServer(app);
const PORT = 5002;

// Socket.io setup
const io = new Server(server, {
  cors: {
    origin: "http://localhost:3000",
    methods: ["GET", "POST"]
  }
});

// Middleware
app.use(cors());
app.use(express.json());

// Indian water sources data
const indianWaterSources = [
  { id: 1, name: "Ganges River", lat: 25.3176, lng: 83.0059, type: "river", region: "Uttar Pradesh" },
  { id: 2, name: "Yamuna River", lat: 25.4230, lng: 81.8833, type: "river", region: "Uttar Pradesh" },
  { id: 3, name: "Brahmaputra", lat: 26.7580, lng: 92.1032, type: "river", region: "Assam" },
  { id: 4, name: "Chennai Water Supply", lat: 13.0827, lng: 80.2707, type: "reservoir", region: "Tamil Nadu" },
  { id: 5, name: "Bangalore Lakes", lat: 12.9716, lng: 77.5946, type: "lake", region: "Karnataka" },
  { id: 6, name: "Mumbai Water Works", lat: 19.0760, lng: 72.8777, type: "treatment", region: "Maharashtra" },
  { id: 7, name: "Kerala Backwaters", lat: 9.4981, lng: 76.3388, type: "backwater", region: "Kerala" },
  { id: 8, name: "Delhi Jal Board", lat: 28.6139, lng: 77.2090, type: "treatment", region: "Delhi" }
];

// Basic route
app.get('/', (req, res) => {
  res.json({ 
    message: 'AquaGuard API Server is running!',
    features: ['IoT Monitoring', 'Real-time Alerts', 'AI Predictions', 'AR/VR Education', 'Gamified Learning']
  });
});

// Socket.io connection handling
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);
  
  // Send initial data
  socket.emit('waterSources', indianWaterSources);
  
  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Function to emit sensor data
function emitSensorData(data) {
  io.emit('sensorData', data);
  
  // Check for anomalies and emit alerts
  const anomalies = [];
  if (data.pH < 6.5 || data.pH > 8.5) anomalies.push('pH');
  if (data.turbidity > 10) anomalies.push('turbidity');
  if (data.tds > 500) anomalies.push('tds');
  
  if (anomalies.length > 0) {
    const alert = {
      ...data,
      timestamp: new Date(),
      message: 'Water quality issues detected: ' + anomalies.join(', '),
      severity: anomalies.length > 1 ? 'high' : 'medium'
    };
    addAlert(alert);
    io.emit('alert', alert);
  }
}

// Simulate IoT data from Indian water sources
setInterval(() => {
  const source = indianWaterSources[Math.floor(Math.random() * indianWaterSources.length)];
  
  const data = {
    deviceId: 'device_' + source.id,
    pH: parseFloat((6.0 + Math.random() * 3).toFixed(2)),
    turbidity: parseFloat((Math.random() * 40).toFixed(2)),
    tds: Math.floor(50 + Math.random() * 1000),
    temperature: parseFloat((20 + Math.random() * 15).toFixed(2)),
    location: {
      lat: source.lat + (Math.random() * 0.02 - 0.01),
      lng: source.lng + (Math.random() * 0.02 - 0.01)
    },
    source: source.name,
    region: source.region,
    type: source.type,
    timestamp: new Date()
  };
  
  // Monsoon effect simulation (June-September)
  const isMonsoon = new Date().getMonth() >= 5 && new Date().getMonth() <= 8;
  if (isMonsoon) {
    data.turbidity = parseFloat((10 + Math.random() * 30).toFixed(2));
    data.tds = Math.floor(300 + Math.random() * 700);
  }
  
  console.log('Generated data from:', source.name);
  addIOTData(data);
  emitSensorData(data);
  
}, 3000);

// Register AI routes
app.use(aiRoutes);

// API Routes
app.get('/api/data', (req, res) => {
  try {
    const data = getRecentData(100);
    res.json(data);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

app.get('/api/water-sources', (req, res) => {
  res.json(indianWaterSources);
});

app.post('/api/sensor-data', (req, res) => {
  try {
    const data = req.body;
    console.log('Received HTTP data:', data);
    
    addIOTData(data);
    emitSensorData(data);
    
    res.json({ success: true, message: 'Data received' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

app.get('/api/alerts', (req, res) => {
  try {
    const alerts = getAlerts();
    res.json(alerts);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

app.post('/api/alerts', (req, res) => {
  try {
    const alert = req.body;
    addAlert(alert);
    io.emit('alert', alert);
    res.json({ success: true, message: 'Alert created' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// AI Prediction Endpoint with Indian context
app.get('/api/predict', (req, res) => {
  try {
    const currentMonth = new Date().getMonth();
    const isMonsoon = currentMonth >= 5 && currentMonth <= 8;
    const isSummer = currentMonth >= 2 && currentMonth <= 5;
    
    let riskLevel = 'low';
    let factors = [];
    
    if (isMonsoon) {
      riskLevel = 'high';
      factors = ['monsoon_season', 'increased_runoff', 'higher_contamination_risk'];
    } else if (isSummer) {
      riskLevel = 'medium';
      factors = ['summer_heat', 'water_scarcity', 'concentration_of_contaminants'];
    }
    
    const prediction = {
      riskLevel: riskLevel,
      confidence: isMonsoon ? 0.85 : 0.65,
      factors: factors,
      recommendation: isMonsoon ? 
        'Increase monitoring frequency during monsoon season' : 
        'Maintain regular monitoring schedule'
    };
    
    res.json(prediction);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// AR/VR educational content
app.get('/api/education/content', (req, res) => {
  const content = {
    modules: [
      {
        id: 1,
        title: "Water Purification Process",
        type: "ar",
        description: "Interactive AR experience showing water treatment steps",
        duration: "5 minutes",
        difficulty: "beginner"
      },
      {
        id: 2,
        title: "Contamination Detection",
        type: "vr", 
        description: "VR simulation of water testing procedures",
        duration: "8 minutes",
        difficulty: "intermediate"
      },
      {
        id: 3,
        title: "Monsoon Impact Simulation",
        type: "vr",
        description: "Experience how monsoon affects water quality",
        duration: "10 minutes",
        difficulty: "advanced"
      }
    ]
  };
  res.json(content);
});

// Gamification endpoints
app.get('/api/games/water-safety-quiz', (req, res) => {
  const quiz = {
    title: "Water Safety Quiz",
    questions: [
      {
        question: "What is the ideal pH range for drinking water?",
        options: ["4.0-5.0", "6.5-8.5", "9.0-10.0", "Any pH is fine"],
        correctAnswer: 1
      },
      {
        question: "Which season typically has the highest water contamination risk in India?",
        options: ["Winter", "Summer", "Monsoon", "Spring"],
        correctAnswer: 2
      },
      {
        question: "What does high turbidity indicate in water?",
        options: ["High mineral content", "Cloudiness/particles", "High temperature", "Low oxygen"],
        correctAnswer: 1
      }
    ]
  };
  res.json(quiz);
});

app.post('/api/games/quiz-score', (req, res) => {
  const { score, total } = req.body;
  console.log('Quiz score: ' + score + '/' + total);
  res.json({ 
    success: true, 
    message: 'You scored ' + score + ' out of ' + total,
    badge: score >= total * 0.8 ? 'Water Safety Expert' : 'Water Learner'
  });
});

// Start server
server.listen(PORT, () => {
  console.log('Server running on port ' + PORT);
  console.log('API: http://localhost:' + PORT + '/api/data');
  console.log('Indian water sources monitoring enabled');
  console.log('AR/VR educational content available');
  console.log('Gamification modules ready');
});

