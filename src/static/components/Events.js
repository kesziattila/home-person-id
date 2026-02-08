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

            <!-- Active filters bar -->
            <div v-if="hasActiveFilters" class="mb-4 flex flex-wrap items-center gap-2 bg-gray-800 rounded-lg px-4 py-3">
                <span class="text-xs text-gray-400 uppercase font-semibold mr-1">Filters:</span>
                <span v-if="filters.event_type" class="inline-flex items-center gap-1 bg-blue-600/20 text-blue-400 text-xs px-2 py-1 rounded-full">
                    Type: {{ filters.event_type }}
                    <button @click="clearFilter('event_type')" class="hover:text-white ml-1">&times;</button>
                </span>
                <span v-if="filters.camera_id" class="inline-flex items-center gap-1 bg-blue-600/20 text-blue-400 text-xs px-2 py-1 rounded-full">
                    Camera: {{ filters.camera_id }}
                    <button @click="clearFilter('camera_id')" class="hover:text-white ml-1">&times;</button>
                </span>
                <span v-if="filters.person_name" class="inline-flex items-center gap-1 bg-blue-600/20 text-blue-400 text-xs px-2 py-1 rounded-full">
                    Person: {{ filters.person_name }}
                    <button @click="clearFilter('person_name')" class="hover:text-white ml-1">&times;</button>
                </span>
                <span v-if="filters.track_id" class="inline-flex items-center gap-1 bg-blue-600/20 text-blue-400 text-xs px-2 py-1 rounded-full">
                    Track: {{ filters.track_id }}
                    <button @click="clearFilter('track_id')" class="hover:text-white ml-1">&times;</button>
                </span>
                <button @click="clearAllFilters" class="text-xs text-gray-500 hover:text-gray-300 ml-2 underline">Clear all</button>
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
                        <tr v-else-if="filteredEvents.length === 0">
                            <td colspan="6" class="px-4 py-8 text-center text-gray-500">No events found in this time range.</td>
                        </tr>
                        <tr v-for="event in filteredEvents" :key="event.id" @click="showEventDetail(event)" class="hover:bg-gray-700 cursor-pointer">
                            <td class="px-4 py-3 text-sm">{{ new Date(event.timestamp).toLocaleString() }}</td>
                            <td class="px-4 py-3 text-sm">
                                <span @click.stop="setFilter('event_type', event.event_type)" class="hover:text-blue-400 hover:underline cursor-pointer">{{ event.event_type }}</span>
                            </td>
                            <td class="px-4 py-3 text-sm">
                                <span @click.stop="setFilter('camera_id', event.camera_id)" class="hover:text-blue-400 hover:underline cursor-pointer">{{ event.camera_id }}</span>
                            </td>
                            <td class="px-4 py-3 text-sm">
                                <span v-if="event.person_name" @click.stop="setFilter('person_name', event.person_name)" class="hover:text-blue-400 hover:underline cursor-pointer">{{ event.person_name }}</span>
                                <span v-else>-</span>
                            </td>
                            <td class="px-4 py-3 text-sm">{{ event.confidence ? (event.confidence * 100).toFixed(0) + '%' : '-' }}</td>
                            <td class="px-4 py-3 text-sm font-mono">
                                <span v-if="event.track_id" @click.stop="setFilter('track_id', event.track_id)" class="hover:text-blue-400 hover:underline cursor-pointer">{{ event.track_id }}</span>
                                <span v-else>-</span>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    `,
    props: {
        autoRefresh: Boolean,
        applyFilter: {
            type: Object,
            default: null
        }
    },
    emits: ['show-event-detail', 'filter-applied'],
    data() {
        return {
            events: [],
            loading: true,
            error: null,
            hours: '24',
            refreshInterval: null,
            filters: {
                event_type: null,
                camera_id: null,
                person_name: null,
                track_id: null,
            },
        };
    },
    computed: {
        hasActiveFilters() {
            return Object.values(this.filters).some(v => v !== null);
        },
        filteredEvents() {
            let result = this.events;
            if (this.filters.person_name) {
                result = result.filter(e => e.person_name === this.filters.person_name);
            }
            return result;
        }
    },
    async created() {
        await this.loadEvents();
        if (this.autoRefresh) {
            this.refreshInterval = setInterval(this.loadEvents, 10000);
        }
        this._onKeydown = (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
            if (e.key === 'r' || e.key === 'R') { this.loadEvents(); }
            if (e.key === 'c' || e.key === 'C') { this.clearAllFilters(); }
        };
        window.addEventListener('keydown', this._onKeydown);
    },
    beforeUnmount() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        window.removeEventListener('keydown', this._onKeydown);
    },
    watch: {
        autoRefresh(newVal) {
            if (newVal && !this.refreshInterval) {
                this.refreshInterval = setInterval(this.loadEvents, 10000);
            } else if (!newVal && this.refreshInterval) {
                clearInterval(this.refreshInterval);
                this.refreshInterval = null;
            }
        },
        applyFilter(newFilter) {
            if (newFilter && newFilter.field && newFilter.value) {
                this.filters[newFilter.field] = newFilter.value;
                this.loadEvents();
                this.$emit('filter-applied');
            }
        }
    },
    methods: {
        async loadEvents() {
            this.loading = true;
            this.error = null;
            try {
                const params = new URLSearchParams();
                params.set('since_hours', this.hours);
                params.set('limit', '100');
                if (this.filters.event_type) params.set('event_type', this.filters.event_type);
                if (this.filters.camera_id) params.set('camera_id', this.filters.camera_id);
                if (this.filters.track_id) params.set('track_id', this.filters.track_id);

                const response = await fetch(`/api/v1/events?${params.toString()}`);
                if (!response.ok) throw new Error('Failed to fetch events');
                this.events = await response.json();
            } catch (e) {
                this.error = 'Error loading events.';
                console.error(e);
            } finally {
                this.loading = false;
            }
        },
        setFilter(field, value) {
            if (this.filters[field] === value) {
                this.filters[field] = null;
            } else {
                this.filters[field] = value;
            }
            this.loadEvents();
        },
        clearFilter(field) {
            this.filters[field] = null;
            this.loadEvents();
        },
        clearAllFilters() {
            this.filters.event_type = null;
            this.filters.camera_id = null;
            this.filters.person_name = null;
            this.filters.track_id = null;
            this.loadEvents();
        },
        showEventDetail(event) {
            this.$emit('show-event-detail', event);
        }
    }
};
