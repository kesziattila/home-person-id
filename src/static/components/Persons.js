import PersonCard from './PersonCard.js';

export default {
    components: {
        PersonCard
    },
    template: `
        <div>
            <div class="mb-4 flex justify-between items-center">
                <div>
                    <h2 class="text-2xl font-bold mb-2">Registered Persons</h2>
                    <p class="text-gray-400 text-sm">Manage enrolled persons and their face images</p>
                </div>
                <button @click="showAddPersonModal" class="bg-green-600 hover:bg-green-700 px-4 py-2 rounded text-sm flex items-center gap-2">
                    <span>+</span> Add Person
                </button>
            </div>
            <div v-if="loading" class="flex justify-center items-center p-16">
                <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
            </div>
            <div v-else-if="error" class="text-red-500 text-center py-20">
                {{ error }}
            </div>
            <div v-else-if="persons.length === 0" class="text-center py-20 text-gray-500">
                <p class="mb-4">No persons registered yet</p>
                <button @click="showAddPersonModal" class="bg-green-600 hover:bg-green-700 px-4 py-2 rounded">
                    Add First Person
                </button>
            </div>
            <div v-else id="persons-grid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <person-card v-for="person in persons" :key="person.id" :person="person" @show-detail="showPersonDetail"></person-card>
            </div>
        </div>
    `,
    props: ['autoRefresh'],
    emits: ['show-person-detail', 'show-add-person'],
    data() {
        return {
            persons: [],
            loading: true,
            error: null,
            refreshInterval: null,
        };
    },
    async created() {
        await this.loadPersons();
        if (this.autoRefresh) {
            this.refreshInterval = setInterval(this.loadPersons, 10000);
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
                this.refreshInterval = setInterval(this.loadPersons, 10000);
            } else if (!newVal && this.refreshInterval) {
                clearInterval(this.refreshInterval);
                this.refreshInterval = null;
            }
        }
    },
    methods: {
        async loadPersons() {
            this.loading = true;
            this.error = null;
            try {
                const response = await fetch('/api/v1/persons');
                if (!response.ok) throw new Error('Failed to fetch persons');
                this.persons = await response.json();
            } catch (e) {
                this.error = 'Error loading persons.';
                console.error(e);
            } finally {
                this.loading = false;
            }
        },
        showAddPersonModal() {
            this.$emit('show-add-person');
        },
        showPersonDetail(personId) {
            this.$emit('show-person-detail', personId);
        }
    }
};
