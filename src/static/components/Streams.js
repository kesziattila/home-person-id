export default {
    template: `
        <div>
            <div v-if="loading" class="flex justify-center items-center p-16">
                <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
            </div>
            <div v-else-if="error" class="text-red-500 text-center py-20">
                {{ error }}
            </div>
            <div v-else-if="cameras.length === 0" class="text-center py-20 text-gray-500">
                No active camera streams found. Waiting for frames...
            </div>
            <div v-else id="camera-grid" class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div v-for="camera in cameras" :key="camera" class="space-y-2">
                    <div class="flex justify-between items-center">
                        <h2 class="text-lg font-semibold px-1">{{ camera }}</h2>
                    </div>
                    <div class="stream-container shadow-2xl">
                        <img :src="'/api/v1/stream/' + camera"
                             class="stream-img"
                             :alt="'Stream for ' + camera"
                             onerror="this.src='https://via.placeholder.com/640x360?text=No+Stream+Available'">
                    </div>
                </div>
            </div>
        </div>
    `,
    props: ['autoRefresh'],
    data() {
        return {
            cameras: [],
            loading: true,
            error: null,
        };
    },
    async created() {
        await this.loadCameras();
    },
    methods: {
        async loadCameras() {
            this.loading = true;
            this.error = null;
            try {
                const response = await fetch('/api/v1/cameras');
                if (!response.ok) throw new Error('Failed to fetch cameras');
                const data = await response.json();
                this.cameras = data.cameras || [];
            } catch (e) {
                this.error = 'Error loading camera streams.';
                console.error(e);
            } finally {
                this.loading = false;
            }
        }
    },
    beforeUnmount() {
        // When the component is destroyed (tab is switched), explicitly stop the streams.
        const streams = this.$el.querySelectorAll('.stream-img');
        streams.forEach(img => {
            img.src = '';
        });
    }
};
