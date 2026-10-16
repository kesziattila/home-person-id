import { formatRelativeTime, parseUtcDate } from '../utils/formatters.js';

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
                <div v-else-if="segments.length === 0" class="text-center py-12 text-gray-500">
                    No movement recorded in this period
                </div>
                <div v-else class="relative pl-6 border-l-2 border-gray-600 space-y-4">
                    <div v-for="(s, i) in segments" :key="i" class="relative">
                        <div class="absolute -left-[1.65rem] top-1 w-3 h-3 rounded-full border-2"
                             :class="s.is_current ? 'bg-green-500 border-green-500'
                                     : (s.is_estimated ? 'bg-gray-800 border-amber-500' : 'bg-gray-500 border-gray-500')"></div>
                        <div class="rounded-lg p-3 flex gap-3"
                             :class="s.is_estimated ? 'bg-gray-800 border border-dashed border-amber-600/60 opacity-80'
                                     : 'bg-gray-750 border border-gray-700'">
                            <img v-if="s.snapshot_event_id" :src="'/api/v1/events/' + s.snapshot_event_id + '/snapshot'"
                                 loading="lazy" @error="$event.target.style.display='none'"
                                 class="w-14 h-20 object-cover object-top rounded flex-shrink-0">
                            <div class="flex-1 min-w-0">
                                <div class="flex justify-between items-start mb-1 gap-2">
                                    <span class="font-semibold text-sm">
                                        {{ s.zone || s.camera_id }}
                                        <span v-if="s.is_estimated"
                                              class="ml-1 text-[0.65rem] uppercase tracking-wide text-amber-400 align-middle">~ estimated</span>
                                    </span>
                                    <span class="text-xs text-gray-500 whitespace-nowrap">{{ formatTime(s.entered_at) }}</span>
                                </div>
                                <!-- Duration bar: width proportional (log-scaled) to time spent -->
                                <div class="mt-1">
                                    <div class="h-1.5 rounded-full bg-gray-700 overflow-hidden">
                                        <div class="h-full rounded-full"
                                             :class="s.is_current ? 'bg-green-500'
                                                     : (s.is_estimated ? 'bg-amber-500/70' : 'bg-blue-500')"
                                             :style="{ width: barWidth(s) + '%' }"></div>
                                    </div>
                                    <div class="flex justify-between items-center mt-0.5">
                                        <span class="text-xs text-gray-400">
                                            {{ s.duration_sec != null ? formatDuration(s.duration_sec) : '—' }}
                                        </span>
                                        <span v-if="s.is_current" class="text-xs text-green-400">Currently here</span>
                                    </div>
                                </div>
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
            segments: [],
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
    computed: {
        maxDuration() {
            return this.segments.reduce(
                (m, s) => Math.max(m, this.effDuration(s)), 0
            );
        }
    },
    methods: {
        close() {
            this.$emit('close');
        },
        effDuration(s) {
            // Finished stay -> its duration; open/current stay -> elapsed so far.
            if (s.duration_sec != null) return s.duration_sec;
            const elapsed = (Date.now() - parseUtcDate(s.entered_at).getTime()) / 1000;
            return elapsed > 0 ? elapsed : 0;
        },
        barWidth(s) {
            // Linear scale: bar length is proportional to time spent. A short stay
            // next to a long one is intentionally a thin sliver (min 1% to stay visible).
            const max = this.maxDuration;
            if (max <= 0) return 1;
            return Math.max(1, Math.round((this.effDuration(s) / max) * 100));
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
                this.segments = data.segments;
            } catch (e) {
                this.error = e.message;
            } finally {
                this.loading = false;
            }
        },
        formatTime(isoString) {
            if (!isoString) return '';
            return parseUtcDate(isoString).toLocaleString(undefined, {
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
