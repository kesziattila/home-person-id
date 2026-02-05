import App from './components/App.js';
import Streams from './components/Streams.js';
import Persons from './components/Persons.js';
import Events from './components/Events.js';
import Tracks from './components/Tracks.js';
import Unidentified from './components/Unidentified.js';

const { createApp } = Vue;

const app = createApp(App);

// Register all the components
app.component('Streams', Streams);
app.component('Persons', Persons);
app.component('Events', Events);
app.component('Tracks', Tracks);
app.component('Unidentified', Unidentified);

app.mount('#app');
