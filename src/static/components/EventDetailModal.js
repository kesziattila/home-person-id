import { formatRelativeTime } from '../utils/formatters.js';

export default {
    template: `
        <div v-if="event" class="modal active fixed inset-0 bg-black bg-opacity-75 items-center justify-center z-50 p-4">
            <div class="bg-gray-800 rounded-lg p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
                <div class="flex justify-between items-start mb-4">
                    <div>
                        <h3 class="text-xl font-bold">{{ event.event_type }} - {{ event.camera_id }}</h3>
                        <p class="text-gray-400 text-sm">{{ new Date(event.timestamp).toLocaleString() }}</p>
                    </div>
                    <button @click="close" class="text-gray-400 hover:text-white text-2xl">&times;</button>
                </div>
                
                <div class="space-y-6" v-html="eventContent"></div>

                <div class="mt-6 pt-4 border-t border-gray-700">
                    <button @click="close" class="w-full bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded">
                        Close
                    </button>
                </div>
            </div>
        </div>
    `,
    props: {
        event: {
            type: Object,
            default: null
        }
    },
    emits: ['close'],
    computed: {
        eventContent() {
            if (!this.event) return '';

            const ev = this.event;
            const extra = ev.extra_data ? JSON.stringify(ev.extra_data, null, 2) : '{}';
            const personDisplay = ev.person_name || (ev.person_id ? `ID: ${ev.person_id}` : 'Unidentified');
            const confidenceDisplay = (ev.confidence !== null && ev.confidence !== undefined)
                ? (ev.confidence * 100).toFixed(1) + '%' : 'N/A';

            let snapshotHtml = `<p class="text-sm text-gray-500 italic">No snapshot available</p>`;
            if (ev.snapshot_path) {
                snapshotHtml = `
                    <div class="space-y-2">
                        <img src="/api/v1/events/${ev.id}/snapshot" 
                             class="max-w-full h-auto rounded-lg shadow-2xl mx-auto border border-gray-700" 
                             alt="Event Snapshot">
                    </div>
                `;
            }

            return `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div class="space-y-6">
                        <section>
                            <h4 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Details</h4>
                            <div class="bg-gray-700/50 rounded-lg p-4 space-y-3">
                                <div class="flex justify-between">
                                    <span class="text-gray-400">Person</span>
                                    <span class="font-medium text-blue-400">${personDisplay}</span>
                                </div>
                                <div class="flex justify-between">
                                    <span class="text-gray-400">Confidence</span>
                                    <span class="font-medium">${confidenceDisplay}</span>
                                </div>
                            </div>
                        </section>
                        <section>
                            <h4 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Metadata</h4>
                            <pre class="bg-gray-900 text-green-400 p-4 rounded-lg text-xs overflow-x-auto border border-gray-700 font-mono">${this.escapeHtml(extra)}</pre>
                        </section>
                    </div>
                    <div class="space-y-4">
                        <h4 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Snapshot</h4>
                        ${snapshotHtml}
                    </div>
                </div>
            `;
        }
    },
    methods: {
        close() {
            this.$emit('close');
        },
        escapeHtml(str) {
            return String(str)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;');
        }
    }
};
