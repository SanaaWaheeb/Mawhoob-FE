import {Component, OnInit} from '@angular/core';
import {ActivatedRoute, Router} from "@angular/router";

@Component({
    selector: 'app-sign-up',
    standalone: false,
    templateUrl: './sign-up.component.html',
    styleUrl: './sign-up.component.scss'
})
export class SignUpComponent implements OnInit {
    typeSignUp: string = ''

    constructor(
        private route: ActivatedRoute,
        private router: Router
    ) {
    }

    ngOnInit() {
        this.getDataFromRoute();
    }

    getDataFromRoute() {
        this.route.params.subscribe(params => {
            this.typeSignUp = params['type'];
        })
    }

    onNavigateMain() {
        this.router.navigate(['main', 'home']);
    }
}
