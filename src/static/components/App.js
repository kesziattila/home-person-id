import Streams from './Streams.js';
import Persons from './Persons.js';
import Events from './Events.js';
import Tracks from './Tracks.js';
import Unidentified from './Unidentified.js';
import PersonDetailModal from './PersonDetailModal.js';
import AssignFaceModal from './AssignFaceModal.js';
import EventDetailModal from './EventDetailModal.js';

export default {
    components: {
        Streams,
        Persons,
        Events,
        Tracks,
        Unidentified,
        PersonDetailModal,
        AssignFaceModal,
        EventDetailModal,
    },
    template: `
        <div>
            <nav class="bg-gray-800 p-4 shadow-lg">
                <div class="container mx-auto flex justify-between items-center">
                    <h1 class="text-xl font-bold">Home Person ID</h1>
                    <div id="status-badge" class="px-3 py-1 rounded-full text-xs font-medium bg-green-500 text-green-950">
                        System Online
                    </div>
                    <div class="flex items-center gap-2 ml-4">
                        <input type="checkbox" id="auto-refresh-check" v-model="autoRefresh" class="rounded border-gray-600 bg-gray-700">
                        <label for="auto-refresh-check" class="text-xs text-gray-400 cursor-pointer select-none">Auto Refresh</label>
                    </div>
                </div>
            </nav>

            <div class="bg-gray-800 border-b border-gray-700">
                <div class="container mx-auto flex space-x-2">
                    <button v-for="tab in tabs" :key="tab.id" 
                            @click="activeTab = tab.id"
                            :class="['tab-button', { 'active': activeTab === tab.id }]">
                        {{ tab.name }}
                    </button>
                </div>
            </div>

            <main class="container mx-auto p-6">
                <component :is="activeTabComponent" 
                           :auto-refresh="autoRefresh"
                           @show-person-detail="showPersonDetail"
                           @show-assign-face="showAssignFace"
                           @show-event-detail="showEventDetail"
                           @face-assigned="handleFaceAssigned"
                           @person-deleted="handlePersonDeleted">
                </component>
            </main>

            <!-- Modals -->
            <person-detail-modal :person-id="selectedPersonId" 
                                 @close="selectedPersonId = null"
                                 @person-deleted="handlePersonDeleted"
                                 @preview-image="showImagePreview">
            </person-detail-modal>

            <assign-face-modal :face-id="selectedFaceId"
                               @close="selectedFaceId = null"
                               @face-assigned="handleFaceAssigned">
            </assign-face-modal>

            <event-detail-modal :event="selectedEvent"
                                @close="selectedEvent = null">
            </event-detail-modal>

        </div>
    `,
    data() {
        return {
            activeTab: 'streams',
            autoRefresh: true,
            tabs: [
                { id: 'streams', name: 'Live Streams' },
                { id: 'persons', name: 'Persons' },
                { id: 'unidentified', name: 'Unidentified' },
                { id: 'events', name: 'Events' },
                { id: 'tracks', name: 'Active Tracks' },
            ],
            selectedPersonId: null,
            selectedFaceId: null,
            selectedImageId: null,
            selectedEvent: null,
        };
    },
    computed: {
        activeTabComponent() {
            const tabExists = this.tabs.some(tab => tab.id === this.activeTab);
            if (!tabExists) return null;
            return this.activeTab.charAt(0).toUpperCase() + this.activeTab.slice(1);
        }
    },
    watch: {
        activeTab(newTab) {
            if (window.location.hash !== '#' + newTab) {
                window.location.hash = newTab;
            }
        }
    },
    created() {
        const hash = window.location.hash.replace('#', '');
        if (this.tabs.some(tab => tab.id === hash)) {
            this.activeTab = hash;
        }

        window.addEventListener('hashchange', () => {
            const newHash = window.location.hash.replace('#', '');
            if (this.tabs.some(tab => tab.id === newHash)) {
                this.activeTab = newHash;
            }
        });
    },
    methods: {
        showPersonDetail(personId) {
            this.selectedPersonId = personId;
        },
        showAssignFace(faceId) {
            this.selectedFaceId = faceId;
        },
        showEventDetail(event) {
            this.selectedEvent = event;
        },
        showImagePreview(imageId) {
            this.selectedImageId = imageId;
            // Logic to show image preview modal will go here
            console.log("Previewing image:", imageId);
        },
        handlePersonDeleted() {
            if (this.activeTab === 'persons') {
                this.forceRerender(this.activeTab);
            }
        },
        handleFaceAssigned() {
            if (this.activeTab === 'unidentified') {
                this.forceRerender(this.activeTab);
            }
        },
        forceRerender(tabName) {
            this.activeTab = '';
            this.$nextTick(() => {
                this.activeTab = tabName;
            });
        }
    }
};
