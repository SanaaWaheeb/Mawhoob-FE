import {Component} from '@angular/core';
import {Router} from "@angular/router";

@Component({
    selector: 'app-display-auth',
    standalone: false,
    templateUrl: './display-auth.component.html',
    styleUrl: './display-auth.component.scss'
})
export class DisplayAuthComponent {
    constructor(private router: Router) {
    }

    onNavigateToSignIn() {
        this.router.navigate(['/sign-in']);
    }
}
