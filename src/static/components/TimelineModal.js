import { formatRelativeTime } from '../utils/formatters.js';

export default {
    template: `
        <div v-if="visible" @click.self="close" class="modal active fixed inset-0 bg-black bg-opacity-75 items-center justify-center z-50 p-4">
            <div class="bg-gray-800 rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
                <div class="flex justify-between items-start mb-4">
                    <div>
                        <h3 class="text-xl font-bold">Movement Timeline</h3>
                        <p class="text-gray-400 text-sm">{{ personName }}</p>
                    </div>
                    <button @click="close" class="text-gray-400 hover:text-white text-2xl">&times;</button>
                </div>

                <div class="mb-4">
                    <select v-model="sinceHours" @change="loadTimeline" class="bg-gray-700 border border-gray-600 rounded px-3 py-1 text-sm">
                        <option :value="1">Last hour</option>
                        <option :value="6">Last 6 hours</option>
                        <option :value="24">Last 24 hours</option>
                        <option :value="72">Last 3 days</option>
                        <option :value="168">Last 7 days</option>
                    </select>
                </div>

                <div v-if="loading" class="flex justify-center items-center p-12">
                    <div class="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500"></div>
                </div>
                <div v-else-if="error" class="text-red-500 text-center py-8">{{ error }}</div>
                <div v-else-if="sightings.length === 0" class="text-center py-12 text-gray-500">
                    No movement recorded in this period
                </div>
                <div v-else class="relative pl-6 border-l-2 border-gray-600 space-y-4">
                    <div v-for="(s, i) in sightings" :key="i" class="relative">
                        <div class="absolute -left-[1.65rem] top-1 w-3 h-3 rounded-full"
                             :class="s.exited_at ? 'bg-gray-500' : 'bg-green-500'"></div>
                        <div class="bg-gray-750 rounded-lg p-3 border border-gray-700">
                            <div class="flex justify-between items-start mb-1">
                                <span class="font-semibold text-sm">{{ s.entry_zone || s.camera_id }}</span>
                                <span class="text-xs text-gray-500">{{ formatTime(s.entered_at) }}</span>
                            </div>
                            <div class="text-xs text-gray-400 space-y-0.5">
                                <div>Camera: {{ s.camera_id }}</div>
                                <div v-if="s.entry_zone">Entered: {{ s.entry_zone }}</div>
                                <div v-if="s.exit_zone">Exited: {{ s.exit_zone }}</div>
                                <div v-if="s.duration_sec != null">Duration: {{ formatDuration(s.duration_sec) }}</div>
                                <div v-if="!s.exited_at" class="text-green-400">Currently here</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `,
    props: ['personId'],
    emits: ['close'],
    data() {
        return {
            visible: false,
            loading: false,
            error: null,
            personName: '',
            sightings: [],
            sinceHours: 24,
        };
    },
    watch: {
        async personId(newId) {
            if (newId) {
                this.visible = true;
                await this.loadTimeline();
            } else {
                this.visible = false;
            }
        }
    },
    methods: {
        close() {
            this.$emit('close');
        },
        async loadTimeline() {
            if (!this.personId) return;
            this.loading = true;
            this.error = null;
            try {
                const res = await fetch(`/api/v1/persons/${this.personId}/timeline?since_hours=${this.sinceHours}`);
                if (!res.ok) throw new Error('Failed to load timeline');
                const data = await res.json();
                this.personName = data.person_name;
                this.sightings = data.sightings;
            } catch (e) {
                this.error = e.message;
            } finally {
                this.loading = false;
            }
        },
        formatTime(isoString) {
            if (!isoString) return '';
            let utcString = isoString;
            if (!isoString.endsWith('Z') && !isoString.includes('+') && !isoString.includes('-', 10)) {
                utcString = isoString + 'Z';
            }
            return new Date(utcString).toLocaleString(undefined, {
                month: 'short', day: 'numeric',
                hour: '2-digit', minute: '2-digit'
            });
        },
        formatDuration(sec) {
            if (sec < 60) return `${Math.floor(sec)}s`;
            const mins = Math.floor(sec / 60);
            const secs = Math.floor(sec % 60);
            if (mins < 60) return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
            const hours = Math.floor(mins / 60);
            const remMins = mins % 60;
            return remMins > 0 ? `${hours}h ${remMins}m` : `${hours}h`;
        }
    }
};
