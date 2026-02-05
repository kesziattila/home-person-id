import { formatRelativeTime } from '../utils/formatters.js';

export default {
    template: `
        <div @click="$emit('show-detail', person.id)"
             class="bg-gray-800 rounded-lg p-4 border border-gray-700 cursor-pointer hover:border-blue-500 transition-colors">
            <div class="flex gap-4">
                <div class="flex-shrink-0">
                    <img v-if="firstImage" :src="firstImage" class="face-image" :alt="person.name">
                    <div v-else class="face-placeholder">No image</div>
                </div>
                <div class="flex-1 min-w-0">
                    <div class="flex justify-between items-start mb-2">
                        <h3 class="text-lg font-semibold truncate">{{ person.name }}</h3>
                        <span :class="statusColor" class="text-xs px-2 py-1 rounded-full">{{ statusText }}</span>
                    </div>
                    <div class="space-y-1 text-sm text-gray-400">
                        <div>{{ person.face_count }} face(s)</div>
                        <div v-if="location && location.current_camera_id">{{ location.current_camera_id }}</div>
                        <div v-if="location && location.last_seen">{{ formattedLastSeen }}</div>
                    </div>
                </div>
            </div>
        </div>
    `,
    props: ['person'],
    data() {
        return {
            location: null,
            firstImage: null,
        };
    },
    computed: {
        statusColor() {
            if (!this.location) return 'bg-gray-600';
            return this.location.status === 'active' ? 'bg-green-600' : 'bg-gray-600';
        },
        statusText() {
            if (!this.location) return 'Unknown';
            return this.location.status === 'active' ? 'Active' : (this.location.status === 'never_seen' ? 'Never Seen' : 'Inactive');
        },
        formattedLastSeen() {
            return this.location ? formatRelativeTime(this.location.last_seen) : '';
        }
    },
    async created() {
        await this.fetchDetails();
    },
    methods: {
        async fetchDetails() {
            try {
                // Fetch location and first image in parallel
                const [locationRes, detailRes] = await Promise.all([
                    fetch(`/api/v1/persons/${this.person.id}/location`),
                    fetch(`/api/v1/persons/${this.person.id}/detail`)
                ]);
                this.location = await locationRes.json();
                const detail = await detailRes.json();
                if (detail.images && detail.images.length > 0 && detail.images[0].has_image) {
                    this.firstImage = `/api/v1/faces/image/${detail.images[0].id}`;
                }
            } catch (e) {
                console.error(`Failed to fetch details for person ${this.person.id}`, e);
            }
        }
    }
};
