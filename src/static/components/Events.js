import { formatRelativeTime } from '../utils/formatters.js';

export default {
    template: `
        <div>
            <div class="mb-4 flex justify-between items-center">
                <div>
                    <h2 class="text-2xl font-bold mb-2">Recent Events</h2>
                    <p class="text-gray-400 text-sm">Detection and identification events from all cameras</p>
                </div>
                <div class="flex gap-2">
                    <select v-model="hours" @change="loadEvents" class="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">
                        <option value="1">Last 1 hour</option>
                        <option value="6">Last 6 hours</option>
                        <option value="24">Last 24 hours</option>
                        <option value="168">Last 7 days</option>
                    </select>
                    <button @click="loadEvents" class="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded text-sm">
                        Refresh
                    </button>
                </div>
            </div>
            <div class="bg-gray-800 rounded-lg overflow-hidden">
                <table class="w-full">
                    <thead class="bg-gray-700">
                        <tr>
                            <th class="px-4 py-3 text-left text-xs font-medium uppercase">Time</th>
                            <th class="px-4 py-3 text-left text-xs font-medium uppercase">Type</th>
                            <th class="px-4 py-3 text-left text-xs font-medium uppercase">Camera</th>
                            <th class="px-4 py-3 text-left text-xs font-medium uppercase">Person</th>
                            <th class="px-4 py-3 text-left text-xs font-medium uppercase">Confidence</th>
                            <th class="px-4 py-3 text-left text-xs font-medium uppercase">Track</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-gray-700">
                        <tr v-if="loading">
                            <td colspan="6" class="p-4 text-center">
                                <div class="flex justify-center items-center">
                                    <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                                </div>
                            </td>
                        </tr>
                        <tr v-else-if="error">
                            <td colspan="6" class="p-4 text-center text-red-500">{{ error }}</td>
                        </tr>
                        <tr v-else-if="events.length === 0">
                            <td colspan="6" class="px-4 py-8 text-center text-gray-500">No events found in this time range.</td>
                        </tr>
                        <tr v-for="event in events" :key="event.id" @click="showEventDetail(event)" class="hover:bg-gray-700 cursor-pointer">
                            <td class="px-4 py-3 text-sm">{{ new Date(event.timestamp).toLocaleString() }}</td>
                            <td class="px-4 py-3 text-sm">{{ event.event_type }}</td>
                            <td class="px-4 py-3 text-sm">{{ event.camera_id }}</td>
                            <td class="px-4 py-3 text-sm">{{ event.person_name || '-' }}</td>
                            <td class="px-4 py-3 text-sm">{{ event.confidence ? (event.confidence * 100).toFixed(0) + '%' : '-' }}</td>
                            <td class="px-4 py-3 text-sm font-mono">{{ event.track_id ? event.track_id.substring(0, 8) + '...' : '-' }}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    `,
    props: ['autoRefresh'],
    emits: ['show-event-detail'],
    data() {
        return {
            events: [],
            loading: true,
            error: null,
            hours: '24',
            refreshInterval: null,
        };
    },
    async created() {
        await this.loadEvents();
        if (this.autoRefresh) {
            this.refreshInterval = setInterval(this.loadEvents, 10000);
        }
    },
    beforeUnmount() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
    },
    watch: {
        autoRefresh(newVal) {
            if (newVal && !this.refreshInterval) {
                this.refreshInterval = setInterval(this.loadEvents, 10000);
            } else if (!newVal && this.refreshInterval) {
                clearInterval(this.refreshInterval);
                this.refreshInterval = null;
            }
        }
    },
    methods: {
        async loadEvents() {
            this.loading = true;
            this.error = null;
            try {
                const response = await fetch(`/api/v1/events?since_hours=${this.hours}&limit=100`);
                if (!response.ok) throw new Error('Failed to fetch events');
                this.events = await response.json();
            } catch (e) {
                this.error = 'Error loading events.';
                console.error(e);
            } finally {
                this.loading = false;
            }
        },
        showEventDetail(event) {
            this.$emit('show-event-detail', event);
        }
    }
};
