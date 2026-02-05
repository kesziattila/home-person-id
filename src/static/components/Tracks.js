import { formatRelativeTime } from '../utils/formatters.js';

export default {
    template: `
        <div>
            <div class="mb-4">
                <h2 class="text-2xl font-bold mb-2">Active Tracks</h2>
                <p class="text-gray-400 text-sm">Currently tracked persons across all cameras</p>
            </div>
            <div v-if="loading" class="flex justify-center items-center p-16">
                <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
            </div>
            <div v-else-if="error" class="text-red-500 text-center py-20">
                {{ error }}
            </div>
            <div v-else-if="tracks.length === 0" class="text-center py-20 text-gray-500">
                No active tracks
            </div>
            <div v-else id="tracks-grid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div v-for="track in tracks" :key="track.track_id" class="bg-gray-800 rounded-lg p-4 border border-gray-700">
                    <div class="flex justify-between items-start mb-3">
                        <h3 class="text-lg font-semibold">{{ track.person_name || 'Unidentified' }}</h3>
                        <span class="bg-blue-600 text-xs px-2 py-1 rounded-full">Active</span>
                    </div>
                    <div class="space-y-1 text-sm text-gray-400">
                        <div class="font-mono text-xs">{{ track.track_id }}</div>
                        <div>{{ track.camera_id }}</div>
                        <div>{{ Math.floor(track.duration_sec / 60) }}m {{ Math.floor(track.duration_sec % 60) }}s</div>
                        <div>Last seen: {{ formatRelativeTime(track.last_seen) }}</div>
                    </div>
                </div>
            </div>
        </div>
    `,
    props: ['autoRefresh'],
    data() {
        return {
            tracks: [],
            loading: true,
            error: null,
            refreshInterval: null,
        };
    },
    async created() {
        await this.loadTracks();
        if (this.autoRefresh) {
            this.refreshInterval = setInterval(this.loadTracks, 10000);
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
                this.refreshInterval = setInterval(this.loadTracks, 10000);
            } else if (!newVal && this.refreshInterval) {
                clearInterval(this.refreshInterval);
                this.refreshInterval = null;
            }
        }
    },
    methods: {
        formatRelativeTime,
        async loadTracks() {
            this.loading = true;
            this.error = null;
            try {
                const response = await fetch('/api/v1/tracks/active');
                if (!response.ok) throw new Error('Failed to fetch tracks');
                this.tracks = await response.json();
            } catch (e) {
                this.error = 'Error loading active tracks.';
                console.error(e);
            } finally {
                this.loading = false;
            }
        }
    }
};
