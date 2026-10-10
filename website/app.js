const navLinks = document.querySelectorAll(".nav-link");
const sections = document.querySelectorAll(".page-section");
const sidebar = document.querySelector(".sidebar");
const menuToggle = document.querySelector(".menu-toggle");

const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      navLinks.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === `#${entry.target.id}`));
    }
  });
}, { rootMargin: "-30% 0px -60% 0px" });

sections.forEach((section) => observer.observe(section));
navLinks.forEach((link) => link.addEventListener("click", () => sidebar.classList.remove("open")));
menuToggle.addEventListener("click", () => sidebar.classList.toggle("open"));
