

const toggleButton = document.getElementById('toggle-btn');
const sidebar = document.getElementById('sidebar');

function toggleSidebar() {
    sidebar.classList.toggle('close');
    toggleButton.classList.toggle('rotate');

    Array.from(sidebar.getElementsByClassName('show')).forEach(ul => {
       ul.classList.remove('show'); 
       ul.previousElementSibling.classList.remove('rotate');
    });

    closeAllSubMenus();
}

function toggleSubMenu(button) {
    if (!button.nextElementSibling.classList.contains('show')) {
        closeAllSubMenus();
    }

    button.nextElementSibling.classList.toggle('show');
    button.classList.toggle('rotate');

    if (sidebar.classList.contains('close')) {
        sidebar.classList.toggle('close');
        toggleButton.classList.toggle('rotate');
    }
}

function closeAllSubMenus() {
    Array.from(sidebar.getElementsByClassName('show')).forEach(ul => {
       ul.classList.remove('show'); 
       ul.previousElementSibling.classList.remove('rotate');
    });
}




document.querySelectorAll('.faq-question').forEach(item => {
    item.addEventListener('click', () => {
        const answer = item.nextElementSibling;
        const faqItem = item.parentNode;
        const icon = item.querySelector('.faq-icon');

        // Toggle the display of the answer
        answer.style.display = answer.style.display === 'block' ? 'none' : 'block';
        
        // Add/remove active class for rotating icon
        faqItem.classList.toggle('active');
    });
});



